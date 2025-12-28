"""
VLM-based Success Detector
Uses vision-language model to label task success/failure

Based on SOAR paper's approach. We use Qwen2-VL for accessibility on RTX 4070.
"""

import torch
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List


class VLMSuccessDetector:
    """
    Detect task success using VLM visual question answering.
    
    Uses binary question format:
    Q: "Has the robot successfully [task]? Answer yes or no."
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cuda",
        use_4bit: bool = False  # 4-bit may not work on Windows
    ):
        self.device = device
        self.model_name = model_name
        
        print(f"Loading VLM: {model_name}...")
        print("(This may take a few minutes on first run - downloading ~4GB)")
        
        # Import here to avoid slow import at module level
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        # Load model with appropriate precision
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
                print("[OK] Loaded in 4-bit mode")
            except Exception as e:
                print(f"[WARNING] 4-bit loading failed: {e}")
                print("  Falling back to FP16...")
                use_4bit = False
        
        if not use_4bit:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("[OK] Loaded in FP16 mode")
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        
        print(f"[OK] VLM loaded successfully")
        
    def _prepare_image(self, image: np.ndarray | torch.Tensor) -> Image.Image:
        """Convert numpy/torch array to PIL Image."""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Handle different dtypes
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Handle CHW -> HWC if needed
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        return Image.fromarray(image)
    
    @torch.no_grad()
    def detect_success(
        self,
        image: np.ndarray | torch.Tensor,
        task_description: str
    ) -> Tuple[bool, float]:
        """
        Determine if task was successful.
        
        Args:
            image: Final observation image (CHW or HWC format)
            task_description: Natural language task description
            
        Returns:
            success: Boolean indicating success
            confidence: Confidence score [0, 1]
        """
        pil_image = self._prepare_image(image)
        
        # Construct prompt
        prompt = f"""Look at this image of a robot workspace after completing a task.

Task: {task_description}

Question: Did the robot successfully complete this task?
Answer with ONLY 'yes' or 'no'."""
        
        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
        )
        
        # Decode response
        response = self.processor.decode(
            outputs.sequences[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).lower().strip()
        
        # Parse response
        success = "yes" in response
        
        # Estimate confidence from logits
        if outputs.scores:
            first_token_logits = outputs.scores[0][0]
            probs = torch.softmax(first_token_logits, dim=-1)
            confidence = probs.max().item()
        else:
            confidence = 0.5
        
        return success, confidence
    
    @torch.no_grad()
    def propose_tasks(
        self,
        image: np.ndarray | torch.Tensor,
        available_tasks: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Propose feasible tasks given current workspace state.
        
        Used for autonomous practice: VLM suggests what robot can practice.
        
        Args:
            image: Current workspace observation
            available_tasks: List of possible task descriptions
            
        Returns:
            List of (task, feasibility_score) tuples, sorted by score
        """
        pil_image = self._prepare_image(image)
        
        task_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(available_tasks)])
        
        prompt = f"""Look at this robot workspace image.

Available tasks:
{task_list}

Which tasks could the robot attempt right now based on what objects are visible and their positions?
List the task numbers that are feasible, separated by commas."""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )
        
        response = self.processor.decode(
            outputs.sequences[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse response to get task indices
        feasible_tasks = []
        for i, task in enumerate(available_tasks):
            if str(i+1) in response:
                feasible_tasks.append((task, 1.0))
            else:
                feasible_tasks.append((task, 0.0))
        
        # Sort by feasibility
        feasible_tasks.sort(key=lambda x: x[1], reverse=True)
        
        return feasible_tasks


def create_vlm_detector(
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    device: str = "cuda",
    use_4bit: bool = False
) -> VLMSuccessDetector:
    """Factory function to create VLM detector."""
    return VLMSuccessDetector(
        model_name=model_name,
        device=device,
        use_4bit=use_4bit
    )

