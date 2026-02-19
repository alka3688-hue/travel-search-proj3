"""
Ragas Evaluation for Travel Chatbot
Evaluates faithfulness, answer relevancy, context precision, and context recall
"""
import os
import json
import logging
import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

from src.search_engine import TravelSearchEngine
from src.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluation")


class TravelChatbotEvaluator:
    """Evaluates Travel Chatbot using Ragas metrics"""
    
    def __init__(self):
        self.engine = TravelSearchEngine()
        self.golden_dataset_path = Path("data") / "golden_dataset.json"
    
    def load_golden_dataset(self) -> List[Dict]:
        """Load golden dataset for evaluation"""
        if not self.golden_dataset_path.exists():
            logger.warning(f"Golden dataset not found at {self.golden_dataset_path}")
            logger.info("Creating sample golden dataset...")
            return self._create_sample_dataset()
        
        with open(self.golden_dataset_path, 'r') as f:
            dataset = json.load(f)
            logger.info(f"Loaded {len(dataset)} test cases from golden dataset")
            return dataset
    
    def _create_sample_dataset(self) -> List[Dict]:
        """Create sample golden dataset if not exists"""
        sample_data = [
            {
                "question": "What are the baggage allowance rules for international flights?",
                "ground_truth": "For economy class, the baggage allowance is 23 kg (50 lbs) for check-in baggage and 7 kg (15 lbs) for cabin baggage. Business class allows 32 kg (70 lbs) check-in and 12 kg (26 lbs) cabin baggage.",
                "contexts": [
                    "For economy class passengers, check-in baggage allowance is 23 kg (50 lbs) and cabin baggage is 7 kg (15 lbs). Business class passengers are allowed 32 kg (70 lbs) check-in baggage and 12 kg (26 lbs) cabin baggage."
                ],
                "source": "baggage_policy.pdf",
                "category": "air_india_policies"
            },
            {
                "question": "What is Air India's cancellation policy?",
                "ground_truth": "Cancellation charges vary by time before departure: more than 30 days (10% of fare), 15-30 days (25%), 7-14 days (50%), less than 7 days (75%), and no-show (100% - no refund).",
                "contexts": [
                    "Air India cancellation charges: More than 30 days before departure - 10% of fare, 15-30 days - 25%, 7-14 days - 50%, less than 7 days - 75%, no-show - 100% (no refund)."
                ],
                "source": "air-india-coc.pdf",
                "category": "air_india_policies"
            },
            {
                "question": "Do I need a visa to travel from India to UK?",
                "ground_truth": "Yes, Indian citizens need a visa to travel to the UK. You should apply for a UK visitor visa at least 3 months before your travel date.",
                "contexts": [
                    "Indian citizens require a visa to travel to the UK. It is recommended to apply for a UK visitor visa at least 3 months prior to the intended travel date."
                ],
                "source": "visa_requirements.pdf",
                "category": "general"
            },
            {
                "question": "What are the refund policies for flight cancellations?",
                "ground_truth": "Refunds depend on the reason: full refund for airline cancellations, partial refund based on cancellation timing for passenger cancellations, and refunds are processed within 7-21 business days depending on payment method.",
                "contexts": [
                    "Refund policies: Full refund for airline-initiated cancellations. Passenger-initiated cancellations receive partial refunds based on timing. Processing time: 7-21 business days depending on payment method."
                ],
                "source": "refund_policy.pdf",
                "category": "refund_policies"
            },
            {
                "question": "What documents do I need for international travel?",
                "ground_truth": "You need a valid passport (minimum 6 months validity), visa (if required), travel insurance (recommended), COVID-19 vaccination certificate (if applicable), and printed ticket and hotel bookings.",
                "contexts": [
                    "International travel documents required: Valid passport (6 months minimum validity), visa (if applicable), travel insurance (recommended), COVID-19 vaccination certificate (if required), printed tickets and hotel confirmations."
                ],
                "source": "travel_requirements.pdf",
                "category": "general"
            }
        ]
        
        # Save sample dataset
        self.golden_dataset_path.parent.mkdir(exist_ok=True)
        with open(self.golden_dataset_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"Sample golden dataset saved to {self.golden_dataset_path}")
        return sample_data
    
    def generate_responses(self, questions: List[str]) -> tuple:
        """Generate responses for questions"""
        answers = []
        contexts = []
        sources = []
        categories = []
        
        for question in questions:
            logger.info(f"Generating answer for: {question}")
            
            try:
                # Search for relevant documents
                docs, _ = self.engine.search_by_text(question, k=5)
                
                # Generate answer
                answer = self.engine.synthesize_response(docs, question)
                
                # Collect contexts (retrieved documents)
                context_texts = [doc.page_content for doc in docs]
                
                # Collect sources and categories
                doc_sources = [doc.metadata.get('source', 'Unknown') for doc in docs]
                doc_categories = [doc.metadata.get('category', 'unknown') for doc in docs]
                
                answers.append(answer)
                contexts.append(context_texts)
                sources.append(doc_sources)
                categories.append(doc_categories)
                
            except Exception as e:
                logger.error(f"Error generating answer for '{question}': {e}")
                answers.append("Error generating answer")
                contexts.append([])
                sources.append([])
                categories.append([])
        
        return answers, contexts, sources, categories
    
    async def run_ragas_evaluation(self):
        """Run Ragas evaluation"""
        logger.info("=" * 70)
        logger.info("Starting Ragas Evaluation...")
        logger.info("=" * 70)
        
        # Load golden dataset
        golden_data = self.load_golden_dataset()
        
        if not golden_data:
            logger.error("No evaluation data available")
            return None
        
        logger.info(f"Loaded {len(golden_data)} test cases")
        
        # Extract questions and ground truths
        questions = [item["question"] for item in golden_data]
        ground_truths = [[item["ground_truth"]] for item in golden_data]  # Ragas expects list of strings
        expected_sources = [item.get("source", "Unknown") for item in golden_data]
        expected_categories = [item.get("category", "unknown") for item in golden_data]
        
        # Generate answers and contexts
        logger.info("\nGenerating responses...")
        answers, contexts, retrieved_sources, retrieved_categories = self.generate_responses(questions)
        
        # Prepare dataset for Ragas
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        
        # Create HuggingFace Dataset
        hf_dataset = Dataset.from_dict(dataset_dict)
        
        logger.info("\nRunning Ragas metrics...")
        logger.info("Metrics: faithfulness, answer_relevancy, context_precision, context_recall")
        
        # Run evaluation
        try:
            results = evaluate(
                hf_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall
                ],
            )
            
            logger.info("\n" + "=" * 70)
            logger.info("EVALUATION RESULTS")
            logger.info("=" * 70)
            logger.info(f"\nRagas Scores:")
            logger.info(f"  Faithfulness:       {results['faithfulness']:.4f}")
            logger.info(f"  Answer Relevancy:   {results['answer_relevancy']:.4f}")
            logger.info(f"  Context Precision:  {results['context_precision']:.4f}")
            logger.info(f"  Context Recall:     {results['context_recall']:.4f}")
            
            # Calculate category accuracy
            category_matches = sum(
                1 for exp_cat, ret_cats in zip(expected_categories, retrieved_categories)
                if exp_cat in ret_cats
            )
            category_accuracy = category_matches / len(expected_categories) if expected_categories else 0
            logger.info(f"  Category Accuracy:  {category_accuracy:.4f}")
            
            logger.info("=" * 70)
            
            # Save detailed results
            self._save_results(
                results, 
                dataset_dict, 
                expected_sources, 
                expected_categories,
                retrieved_sources,
                retrieved_categories,
                category_accuracy
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            logger.error("Make sure you have OPENAI_API_KEY set for Ragas to work")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    
    def _save_results(
        self, 
        results: dict, 
        dataset_dict: dict,
        expected_sources: List[str],
        expected_categories: List[str],
        retrieved_sources: List[List[str]],
        retrieved_categories: List[List[str]],
        category_accuracy: float
    ):
        """Save evaluation results to file"""
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        
        # Save summary
        summary = {
            "faithfulness": float(results.get('faithfulness', 0)),
            "answer_relevancy": float(results.get('answer_relevancy', 0)),
            "context_precision": float(results.get('context_precision', 0)),
            "context_recall": float(results.get('context_recall', 0)),
            "category_accuracy": float(category_accuracy),
            "total_test_cases": len(dataset_dict["question"]),
            "evaluation_timestamp": pd.Timestamp.now().isoformat()
        }
        
        summary_path = output_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n✅ Evaluation summary saved to {summary_path}")
        
        # Save detailed results with categories
        detailed_df = pd.DataFrame({
            "question": dataset_dict["question"],
            "answer": dataset_dict["answer"],
            "ground_truth": [gt[0] for gt in dataset_dict["ground_truth"]],
            "expected_source": expected_sources,
            "expected_category": expected_categories,
            "retrieved_sources": [", ".join(sources[:3]) for sources in retrieved_sources],
            "retrieved_categories": [", ".join(cats[:3]) for cats in retrieved_categories],
            "contexts": [" | ".join(ctx[:2]) for ctx in dataset_dict["contexts"]]
        })
        
        detailed_path = output_dir / "evaluation_detailed.csv"
        detailed_df.to_csv(detailed_path, index=False)
        
        logger.info(f"✅ Detailed results saved to {detailed_path}")
        
        # Save per-category breakdown
        category_breakdown = {}
        for exp_cat, ret_cats in zip(expected_categories, retrieved_categories):
            if exp_cat not in category_breakdown:
                category_breakdown[exp_cat] = {"total": 0, "matches": 0}
            category_breakdown[exp_cat]["total"] += 1
            if exp_cat in ret_cats:
                category_breakdown[exp_cat]["matches"] += 1
        
        for cat in category_breakdown:
            category_breakdown[cat]["accuracy"] = (
                category_breakdown[cat]["matches"] / category_breakdown[cat]["total"]
            )
        
        breakdown_path = output_dir / "category_breakdown.json"
        with open(breakdown_path, 'w') as f:
            json.dump(category_breakdown, f, indent=2)
        
        logger.info(f"✅ Category breakdown saved to {breakdown_path}")
    
    def run(self):
        """Run evaluation (sync wrapper)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.run_ragas_evaluation())


def run_evaluation():
    """Main evaluation function"""
    evaluator = TravelChatbotEvaluator()
    results = evaluator.run()
    
    if results:
        # Check if evaluation passes minimum thresholds
        min_faithfulness = 0.7
        min_relevancy = 0.7
        
        passed = (
            results.get('faithfulness', 0) >= min_faithfulness and
            results.get('answer_relevancy', 0) >= min_relevancy
        )
        
        if passed:
            logger.info("\n✅ EVALUATION PASSED")
            logger.info(f"   Faithfulness: {results.get('faithfulness', 0):.4f} (>= {min_faithfulness})")
            logger.info(f"   Answer Relevancy: {results.get('answer_relevancy', 0):.4f} (>= {min_relevancy})")
            return 0
        else:
            logger.warning("\n⚠️  EVALUATION BELOW THRESHOLDS")
            logger.warning(f"   Faithfulness: {results.get('faithfulness', 0):.4f} (target: {min_faithfulness})")
            logger.warning(f"   Answer Relevancy: {results.get('answer_relevancy', 0):.4f} (target: {min_relevancy})")
            return 1
    else:
        logger.error("\n❌ EVALUATION FAILED")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = run_evaluation()
    sys.exit(exit_code)