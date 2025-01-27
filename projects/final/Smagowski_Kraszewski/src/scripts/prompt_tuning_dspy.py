import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.datasets import Dataset
from dataclasses import dataclass
import pandas as pd
from typing import Dict, List, Tuple
import os
import logging
from datetime import datetime
from pydantic_settings import BaseSettings, SettingsConfigDict
import random
import numpy as np

# Set fixed seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Set up logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename=f'logs/prompt_tuning_{timestamp}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define signature types for clarity
@dataclass
class Input:
    label: str
    content: str

@dataclass
class Output:
    classification: str
    reasoning: str = ""

class BrainrotDataset(Dataset):
    def __init__(self, data: pd.DataFrame, seed: int = SEED):
        self.data = data
        self.rng = np.random.RandomState(seed)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return Input(label=row['label'], content=row['content']), Output(classification=row['classification'])
    
    def __len__(self):
        return len(self.data)

class ChainOfThoughtFewShotClassifier(dspy.Module):
    def __init__(self, seed: int = SEED):
        super().__init__()
        self.seed = seed
        self.predictor = dspy.Predict("label, content -> classification")
        self.predictor.instructions = """Analyze if webpage content matches the user's learning goal.
        Output ONLY 'True' if the content directly supports or relates to the learning goal.
        Output ONLY 'False' if the content would distract or is unrelated to the learning goal.
        The output must be exactly 'True' or 'False', nothing else."""
    
    def forward(self, label: str, content: str) -> Output:
        prompt = f"""Let's analyze this step by step:
        1. Understand the user's learning objective: {label}
        2. Examine the page content
        3. Identify key educational concepts
        4. Check alignment with learning goal
        5. Consider learning value
        6. Evaluate time-value ratio

        Here are some examples:
        User's Goal: "Learn Python Programming"
        Content: "A comprehensive guide to Python functions..."
        Answer: True (Content directly supports learning objective)

        User's Goal: "Study World War II"
        Content: "Top 10 recipes for chocolate chip cookies..."
        Answer: False (Content would distract from learning objective)"""

        result = self.predictor(label=label, content=content, context=prompt)
        return Output(classification=result.classification, reasoning=result.reasoning)

class FewShotClassifier(dspy.Module):
    def __init__(self, seed: int = SEED):
        super().__init__()
        self.seed = seed
        self.predictor = dspy.Predict("""Determine if content matches learning goal.
                                      Return True if content supports learning, False if it's a distraction.""")
    
    def forward(self, label: str, content: str) -> Output:
        examples = """Examples:
        User's Goal: "Learn Python Programming"
        Content: "A comprehensive guide to Python functions..."
        Answer: True

        User's Goal: "Study World War II"
        Content: "Top 10 recipes for chocolate chip cookies..."
        Answer: False"""

        result = self.predictor(label=label, content=content, context=examples)
        return Output(classification=result.classification)

class RolePromptingClassifier(dspy.Module):
    def __init__(self, seed: int = SEED):
        super().__init__()
        self.seed = seed
        self.predictor = dspy.Predict("""As an expert educational content curator,
        evaluate if content supports learning objective.
        Return True if content supports learning, False if it's a distraction.""")
    
    def forward(self, label: str, content: str) -> Output:
        considerations = """Consider:
        1. Educational value
        2. Relevance to objective
        3. Knowledge advancement potential"""

        result = self.predictor(label=label, content=content, context=considerations)
        return Output(classification=result.classification, reasoning=result.explanation)

class ZeroShotCoTClassifier(dspy.Module):
    def __init__(self, seed: int = SEED):
        super().__init__()
        self.seed = seed
        self.predictor = dspy.Predict("""Evaluate learning opportunity through step-by-step analysis.
        Return True if content supports learning, False if it's a distraction.""")
    
    def forward(self, label: str, content: str) -> Output:
        steps = f"""Let's evaluate:
        1. What is the user trying to learn? ({label})
        2. What knowledge does this content provide?
        3. Would it advance the user's goal?
        4. Could it distract from the objective?
        5. Is this the right time to engage?"""

        result = self.predictor(label=label, content=content, context=steps)
        return Output(classification=result.classification, reasoning=result.steps)

class StructuredReasoningClassifier(dspy.Module):
    def __init__(self, seed: int = SEED):
        super().__init__()
        self.seed = seed
        self.predictor = dspy.Predict("""Analyze content using structured framework.
        Return True if content supports learning, False if it's a distraction.""")
    
    def forward(self, label: str, content: str) -> Output:
        framework = f"""Assessment Framework:
        1. User Objective Analysis:
           - Primary goal: {label}
           - Focus requirement
        2. Content Evaluation:
           - Knowledge contribution
           - Relevance
        3. Decision Factors:
           - Learning benefit
           - Distraction potential
           - Time-value ratio"""

        result = self.predictor(label=label, content=content, context=framework)
        return Output(classification=result.classification, reasoning=result.analysis)

class BrainrotOptimizer:
    def __init__(self, train_data: pd.DataFrame, val_data: pd.DataFrame, seed: int = SEED):
        self.seed = seed
        self.train_dataset = BrainrotDataset(train_data, seed=seed)
        self.val_dataset = BrainrotDataset(val_data, seed=seed)
        
        self.classifiers = {
            # 'chain_of_thought_few_shot': ChainOfThoughtFewShotClassifier(seed=seed),
            # 'few_shot': FewShotClassifier(seed=seed),
            # 'role_prompting': RolePromptingClassifier(seed=seed),
            'zero_shot_cot': ZeroShotCoTClassifier(seed=seed),
            # 'structured_reasoning': StructuredReasoningClassifier(seed=seed)
        }
        self.compiled_classifiers = {}
        self.results = {}

    def metric(self, gold: Output, pred: Output) -> float:
        """Evaluation metric for the teleprompter"""
        return gold.classification.lower() == pred.classification.lower()

    def optimize_classifiers(self):
        """Optimize all classifiers using modern DSPy syntax"""
        # Configure teleprompter
        teleprompter = BootstrapFewShot(
            metric=self.metric,
            max_bootstrapped_demos=8,
            max_labeled_demos=8,
            max_rounds=10,
            seed=self.seed
        )

        for name, classifier in self.classifiers.items():
            logger.info(f"Optimizing {name}...")
            try:
                # Compile program
                compiled = teleprompter.compile(
                    classifier,
                    train_data=self.train_dataset,
                    eval_data=self.val_dataset
                )
                self.compiled_classifiers[name] = compiled

                # Evaluate on validation set
                correct = 0
                total = len(self.val_dataset)
                
                for input_data, gold in self.val_dataset:
                    pred = compiled(input_data.label, input_data.content)
                    if self.metric(gold, pred):
                        correct += 1

                accuracy = correct / total
                self.results[name] = {
                    'accuracy': accuracy,
                    'compiled_model': compiled
                }
                logger.info(f"{name} accuracy: {accuracy:.3f}")

            except Exception as e:
                logger.error(f"Error optimizing {name}: {str(e)}")

        return self.results

    def get_best_classifier(self) -> Tuple[str, dspy.Module]:
        """Return the best performing classifier"""
        if not self.results:
            raise ValueError("No results available. Run optimize_classifiers first.")
        
        best_name = max(self.results.items(), 
                       key=lambda x: x[1]['accuracy'])[0]
        return best_name, self.compiled_classifiers[best_name]

def main():
    class Settings(BaseSettings):
        CRITIC_LLM: str
        OPENAI_API_KEY: str
        model_config = SettingsConfigDict(
            env_file='.env',
            env_file_encoding='utf-8',
            case_sensitive=True
        )
    
    settings = Settings()
    os.environ['OPENAI_API_KEY'] = settings.OPENAI_API_KEY
    # Configure DSPy with modern syntax
    lm = dspy.OpenAI(model='gpt-4o-mini', seed=SEED)
    dspy.settings.configure(lm=lm, trace=True)

    # Load and preprocess the dataset
    df = pd.read_csv('processed-data/ready_dataset.csv')
    
    # Create new dataframe with required format
    processed_data = pd.DataFrame({
        'label': [row.matching_label if row.matching_status else row.non_matching_label 
                 for _, row in df.iterrows()],
        'content': df['markdown_content'].tolist(),
        'classification': df['matching_status'].map({True: "True", False: "False"}).tolist()
    })
    
    # First split into train+val and test sets (80-20 split)
    train_val_size = int(0.8 * len(processed_data))
    
    train_val_data = processed_data.iloc[:train_val_size]
    test_data = processed_data.iloc[train_val_size:]
    
    # Then split train_val into train and validation (70-30 split)
    train_size = int(0.7 * len(train_val_data))
    
    train_data = train_val_data.iloc[:train_size]
    val_data = train_val_data.iloc[train_size:]

    # Create and run optimizer
    optimizer = BrainrotOptimizer(train_data, val_data, seed=SEED)
    results = optimizer.optimize_classifiers()
    
    # Get best classifier
    best_name, best_classifier = optimizer.get_best_classifier()
    logger.info(f"Best performing classifier: {best_name}")
    
    # Log the tuned prompts for all classifiers
    logger.info("Tuned prompts for each classifier:")
    for name, result in optimizer.compiled_classifiers.items():
        logger.info(f"=== {name} ===")
        logger.info("Learned prompt template:")
        logger.info(result.predictor.demos)  # Shows few-shot examples DSPy learned
        logger.info("Final prompt structure:")
        logger.info(result.predictor.instructions)  # Shows the refined instructions
    
    # Evaluate on test set
    correct = 0
    total = len(test_data)
    
    logger.info("Evaluating on test set...")
    for _, row in test_data.iterrows():
        test_input = {
            'label': row['label'],
            'content': row['content']
        }
        prediction = best_classifier(**test_input)
        if prediction == row['classification']:
            correct += 1
            
    test_accuracy = correct / total
    logger.info(f"Test set accuracy: {test_accuracy:.3f}")

if __name__ == "__main__":
    main()