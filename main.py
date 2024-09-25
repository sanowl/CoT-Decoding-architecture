import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional, Dict
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoTDecoder:
    def __init__(self, model_name: str, device: str = 'cuda', max_length: int = 2048):
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.max_length = min(max_length, self.model.config.max_position_embeddings)
            logger.info(f"Model {model_name} loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise

    def decode(self, input_text: str, k: int = 10, max_new_tokens: int = 128) -> List[str]:
        try:
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]
                top_k_logits, top_k_indices = torch.topk(logits, k)
            
            decoded_paths = []
            with ThreadPoolExecutor() as executor:
                futures = []
                for idx in top_k_indices[0]:
                    sequence = torch.cat([input_ids, idx.unsqueeze(0).unsqueeze(0)], dim=-1)
                    futures.append(executor.submit(self._generate_sequence, sequence, max_new_tokens))
                
                for future in as_completed(futures):
                    decoded_paths.append(future.result())
            
            return decoded_paths
        except Exception as e:
            logger.error(f"Error in decode method: {str(e)}")
            raise

    def _generate_sequence(self, sequence: torch.Tensor, max_new_tokens: int) -> str:
        try:
            with torch.no_grad():
                output = self.model.generate(
                    sequence,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.7
                )
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error in _generate_sequence: {str(e)}")
            return ""

    def compute_delta(self, sequence: str) -> float:
        try:
            input_ids = self.tokenizer.encode(sequence, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0]
                probs = F.softmax(logits, dim=-1)
                top_2_probs, _ = torch.topk(probs, 2, dim=-1)
                delta = (top_2_probs[:, 0] - top_2_probs[:, 1]).mean().item()
            return delta
        except Exception as e:
            logger.error(f"Error in compute_delta: {str(e)}")
            return 0.0

    def extract_answer(self, sequence: str, task_type: str = 'general') -> Optional[str]:
        try:
            if task_type == 'math':
                numbers = re.findall(r'\d+(?:\.\d+)?', sequence)
                return numbers[-1] if numbers else None
            elif task_type == 'multiple_choice':
                options = re.findall(r'\b(A|B|C|D|True|False)\b', sequence)
                return options[-1] if options else None
            else:
                patterns = [
                    r"So the answer is\s*(.*)",
                    r"Therefore,\s*(.*)",
                    r"The final answer is\s*(.*)",
                    r"In conclusion,\s*(.*)"
                ]
                for pattern in patterns:
                    match = re.search(pattern, sequence, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
            return None
        except Exception as e:
            logger.error(f"Error in extract_answer: {str(e)}")
            return None

    def cot_decoding(self, input_text: str, k: int = 10, task_type: str = 'general') -> Tuple[Optional[str], Optional[str]]:
        try:
            start_time = time.time()
            decoded_paths = self.decode(input_text, k)
            
            best_path = None
            best_delta = float('-inf')
            all_answers: Dict[str, float] = {}
            
            for path in decoded_paths:
                answer = self.extract_answer(path, task_type)
                if answer is None:
                    continue
                
                delta = self.compute_delta(path)
                all_answers[answer] = all_answers.get(answer, 0) + delta
                
                if delta > best_delta:
                    best_delta = delta
                    best_path = path
            
            best_answer = max(all_answers, key=all_answers.get) if all_answers else None
            
            end_time = time.time()
            logger.info(f"CoT decoding completed in {end_time - start_time:.2f} seconds")
            
            return best_path, best_answer
        except Exception as e:
            logger.error(f"Error in cot_decoding: {str(e)}")
            return None, None

def evaluate_task(decoder: CoTDecoder, task_data: List[Tuple[str, str]], task_type: str = 'general') -> float:
    correct = 0
    total = len(task_data)
    
    for question, true_answer in task_data:
        _, predicted_answer = decoder.cot_decoding(f"Q: {question}\nA:", task_type=task_type)
        if predicted_answer == true_answer:
            correct += 1
        else:
            logger.info(f"Misclassified: Q: {question}, Predicted: {predicted_answer}, True: {true_answer}")
    
    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    model_name = "Mistral-7B"  # Replace with something u like
    decoder = CoTDecoder(model_name)
    
    task_data = [
        ("I have 3 apples, my dad has 2 more apples than me, how many apples do we have in total?", "8"),
        ("If a train travels 120 km in 2 hours, what is its average speed in km/h?", "60"),
    ]
    
    accuracy = evaluate_task(decoder, task_data, task_type='math')
    print(f"Accuracy: {accuracy:.2f}")