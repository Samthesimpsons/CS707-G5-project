from typing import List, Dict
from tqdm import tqdm
from datetime import datetime
from goldfish.index import MemoryIndex
import json
import torch
from pathlib import Path
import gc
import time
from goldfish.paths import WORKSPACE_DIR

def free():
    gc.collect()
    torch.cuda.empty_cache()
    
def get_prompt_text(external_memory, instruction, neighbours=3):
        # get the most similar context from the external memory to this instruction 
        related_context_documents,related_context_keys = external_memory.search_by_similarity(instruction)
        #print(f'RELATED CONTEXT DOCS AND KEYS-------{related_context_documents}, {related_context_keys}')
        related_information=get_related_context(external_memory,related_context_keys,neighbours)
        #print(f'RI---------{related_information}')
        return related_information

def clean_options_text(qa_json: dict):
    """
    If options already start with A./B./C./D., keep them as is.
    Otherwise, add A./B./C./D. prefixes in order.
    """
    prefixes = ["A.", "B.", "C.", "D."]
    options = qa_json.get("options", [])
    if not options:
        return ""

    cleaned = []
    for i, opt in enumerate(options[:4]):
        s = opt.strip()
        if s.startswith(("A.", "B.", "C.", "D.")):
            cleaned.append(s)
        else:
            cleaned.append(f"{prefixes[i]} {s}")
    return "\n".join(cleaned)



    
def build_prompt_text(qa_json: dict, model_name: str, external_memory, neighbours=3) -> str:
        """
            Generates a prompt that is fed into the MLLM during inference
        """
        prompt_parts = []
        #model_name = self.model.model_path.split("/")[-1]
        instruction = "Respond only with the alphabet or number of the correct multiple-choice answer option (e.g., A)." #"Use the video and given subtitles to answer the question. Respond only with the alphabet of the correct multiple-choice answer option (e.g., A)."
        
        
        # Generate questions and answers
        question = qa_json.get("question", "")
        question_id = qa_json.get("question_id", "")
        options_text = clean_options_text(qa_json=qa_json)

        subs = get_prompt_text(external_memory, question, neighbours)
        subs_text = str(subs)
        #print(subs_text)
        prompt_parts.append(            
            f"\n\nVideo Context (Subtitles):\n{subs_text}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer Options:\n{options_text}\n"    
            f"{instruction}\n\n"
        )
        # if "qwen" in model_name.lower():
        #     prompt_parts.append("""\no__think""")

        prompt = "\n\n".join(prompt_parts)
        return prompt, options_text


def extract_video_clips_paths(qa_json: dict, video_dir: str) -> list:
    """
        Output is a list of video clip paths for the given QA JSON
    """
    video_paths = []
    episodes = qa_json.get("episode_span", [])
    for ep in episodes:
        season_id = int(ep[:2])
        clip_id = int(ep[2:])
        clip_path = Path(video_dir) / f"season_{season_id}" / f"episode_{clip_id}.mp4"
        video_paths.append(str(clip_path))
    return video_paths

def generate_batch(
    model,
    qa_pairs: List[Dict],
    qa_pairs_dir: str,
    video_dir: str,
    output_path: str,
    max_new_tokens: int = 128,
    neighbours = 3,
    use_openai_embedding= False,
    caption_type = 'specific',
    batch_size = 8
) -> List[Dict]:
    """
    Run batch inference for QA pairs.

    Args:
        model: object with `.model_path` and `.run_inference(text_prompt, video_paths, max_new_tokens)`
        qa_pairs: list of QA JSON objects
        qa_pairs_dir: directory or filename of QA set
        output_path: where to save results
        max_new_tokens: max new tokens for generation

    Returns:
        List[Dict]: list of results for all QA entries
    """

    if not output_path:
        raise RuntimeError("Output Save Path for Results is Not Specified")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"\n{'='*80}")
    print(f"Batch Job Inference: Initializing --- {now_str}")
    print(f"{'='*80}\n")

    results = []
    model_name = Path(model.model_path).name
    results_files = list(output_dir.glob("*.json"))
    print(f"Existing results :{results_files}")
    for ep_idx, qa_json in enumerate(tqdm(qa_pairs, desc="EpisodicInference Batch Run")):
        ep_start = time.perf_counter()
        ep_results = []
        prompts = []
        videos = []
        episode_id = qa_json.get("episode")
        print(f"Generating Batch Inference - Index: {ep_idx} --- Episode: {episode_id} --- NN: {neighbours}")
        qa_dataset = qa_json.get("qa_dataset", [])
        check_ep_id = f'_{episode_id}_'
        match = any(check_ep_id in f.name for f in results_files)
        #print(match)
        if match:
            print(f'Skipping {episode_id} as it exists!')
        else:
            episode_num = episode_id[-2:] if episode_id[-2] != '0' else episode_id[-1]
            season_num = episode_id[:2] if episode_id[0] != '0' else episode_id[1]
            video_name = f'episode_{episode_num}'
            #print(video_name)
            embedding_root = WORKSPACE_DIR / (
                "open_ai_embedding" if use_openai_embedding else "embedding"
            )
            embedding_path = (
                embedding_root
                / "demo"
                / f"season_{season_num}_{caption_type}"
                / f"{video_name}.pkl"
            )
            external_memory=MemoryIndex(neighbours,use_openai=use_openai_embedding)
            print(f'Embedding path:{embedding_path}')
            if embedding_path.exists():
                print("Loading embeddings from pkl file")
                external_memory.load_embeddings_from_pkl(embedding_path)
            for q_idx, qa_data in enumerate(tqdm(qa_dataset, desc=f"Running QA Dataset - {episode_id}")):
                question_id = qa_data.get("question_id")
                text_prompt, options_text = build_prompt_text(qa_data, model_name, external_memory, neighbours)
                video_paths = [str(Path(video_dir) / f"season_{season_num}" / f"episode_{episode_num}.mp4")]#extract_video_clips_paths(qa_data, video_dir) 
                #print(video_paths)
                # print(f"\n============{question_id}============")
                # print(f"\n{qa_data}\n")
    
                response = {
                    "model": model_name,
                    "question": qa_data.get("question"),
                    "question_id": qa_data.get("question_id"),
                    "question_type": qa_data.get("question_type"),
                    "ground_truth_answer": qa_data.get("answer"),
                    "options_text": options_text,
                    "model_response": "",
                    "execution_time": "",
                    "video_clips": video_paths,
                    "error": "",
                }
                qa_data["result"] = response
                prompts.append(text_prompt)
                videos.append(video_paths)
                ep_results.append(qa_data) 
            free()
            for start in tqdm(range(0, len(prompts), batch_size), desc="Running batched inference"):
                end = min(start + batch_size, len(prompts))
                batch_pairs = list(zip(
                    prompts[start:end],
                    videos[start:end]
                ))
                
                batch_indices = list(range(start, end))
                batch_prompts = [p for p, _ in batch_pairs]
                batch_clips = [v for _, v in batch_pairs]
    
                # === Run batched inference ===
                try:
                    #print(f"Videos to run inference on: {video_paths}")
                    outputs = model.run_inference_batch(
                        text_prompts=batch_prompts,
                        video_paths=batch_clips,
                        max_new_tokens=max_new_tokens
                    )
                    for idx,batch_idx in enumerate(batch_indices):
                        ep_results[batch_idx]["result"]["model_response"] = outputs[idx]
                        
                except Exception as e:
                    for idx,batch_idx in enumerate(batch_indices):
                        ep_results[batch_idx]["result"]["error"] = f"{e}"
                    print(f"❌ Error in question {batch_indices}: {e}")
                    # continue instead of raising to keep batch running
                    continue               
                
                free()
                ep_elapsed = time.perf_counter() - ep_start
                ep_payload = {
                    "execution_time": f"{ep_elapsed:.3f}s",
                    "qa_dataset": ep_results,            
                }
                               
                # if start == 5:
                #     break
            results.append(ep_results)
            #Save per-episode results
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"results_{Path(qa_pairs_dir).name}_{episode_id}_{now_str}.json"
    
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(ep_payload, f, indent=4)
            print(f"✅ Saved results to: {save_path}")
            print(f"Ending Batch Inference - Index: {ep_idx} --- Episode: {episode_id} "
              f"(episode_time={ep_elapsed:.3f}s)")
            free()

    return results


def load_question_answer_pairs(qa_pairs_dir) -> List[Dict]:
    path = Path(qa_pairs_dir)
    entries = []
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data if isinstance(data, list) else [data]
    elif path.is_dir():
        for f in sorted(path.glob("*.json")):
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                entries += data if isinstance(data, list) else [data]
    else:
        raise FileNotFoundError(f"QA file/dir not found: {path}")
    return entries
    
def get_most_related_clips(related_context_keys,neighbours):
    most_related_clips=set()
    for context_key in related_context_keys:
        if len(context_key.split('__'))>1:
            most_related_clips.add(context_key.split('__')[1])
        if len(most_related_clips)==neighbours:
            break
    #assert len(most_related_clips)!=0, f"No related clips found {related_context_keys}"
    return list(most_related_clips)
    
def get_related_context(external_memory,related_context_keys, neighbours):
    related_information=""
    most_related_clips= get_most_related_clips(related_context_keys, neighbours)
    for clip_name in most_related_clips:
        clip_conversation=""
        general_sum=""
        for key in external_memory.documents.keys():
            if clip_name in key and 'caption' in key:
                general_sum="Clip Summary: "+external_memory.documents[key]
            if clip_name in key and 'subtitle' in key:
                clip_conversation="Clip Subtitles: "+external_memory.documents[key]
        related_information+=f"{general_sum},{clip_conversation}\n"
    return related_information   
