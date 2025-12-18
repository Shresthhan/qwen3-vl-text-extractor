"""
Model Performance Evaluation Script
Tests Qwen3-VL on 4 document types
"""

import requests
import time
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Dict

# Configuration
API_URL = "http://localhost:8000/extract-text"

class DocumentEvaluator:
    """Evaluates model performance on different document types"""
    
    def __init__(self):
        self.results = []
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity percentage"""
        text1 = " ".join(text1.lower().split())
        text2 = " ".join(text2.lower().split())
        similarity = SequenceMatcher(None, text1, text2).ratio()
        return similarity * 100
    
    def extract_text_from_image(self, image_path: str) -> Dict:
        """Send image to API and get extracted text"""
        try:
            start_time = time.time()
            
            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/jpeg')}
                data = {'prompt': 'Extract all text from this image'}
                response = requests.post(API_URL, files=files, data=data, timeout=300)
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'extracted_text': result['extracted_text'],
                    'time_taken': duration,
                    'status': 'success'
                }
            else:
                return {
                    'extracted_text': '',
                    'time_taken': duration,
                    'status': 'failed'
                }
                
        except Exception as e:
            return {
                'extracted_text': '',
                'time_taken': 0,
                'status': f'error: {str(e)[:30]}'
            }
    
    def evaluate_single_image(self, test_case: Dict) -> Dict:
        """Evaluate a single test case"""
        print(f"\n📄 Testing: {test_case['document_type']} - {test_case['image_path']}")
        
        result = self.extract_text_from_image(test_case['image_path'])
        
        if result['status'] == 'success' and test_case.get('expected_text'):
            accuracy = self.calculate_similarity(
                result['extracted_text'], 
                test_case['expected_text']
            )
        else:
            accuracy = 0.0
        
        evaluation = {
            'document_type': test_case['document_type'],
            'image_file': Path(test_case['image_path']).name,
            'expected_text': test_case.get('expected_text', 'N/A'),
            'extracted_text': result['extracted_text'],
            'accuracy': round(accuracy, 2),
            'time_taken': round(result['time_taken'], 2),
            'status': '✅ Pass' if accuracy >= 70 else '❌ Fail'
        }
        
        print(f"   Expected: {evaluation['expected_text'][:60]}")
        print(f"   Extracted: {evaluation['extracted_text'][:60]}")
        print(f"   Accuracy: {evaluation['accuracy']}%")
        print(f"   Time: {evaluation['time_taken']}s")
        print(f"   Status: {evaluation['status']}")
        
        self.results.append(evaluation)
        return evaluation
    
    def run_evaluation(self, test_cases: List[Dict]):
        """Run evaluation on all test cases"""
        print("="*70)
        print("🚀 STARTING MODEL EVALUATION")
        print("="*70)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] Processing...")
            self.evaluate_single_image(test_case)
            time.sleep(1)
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE")
        print("="*70)
    
    def print_summary(self):
        """Print summary statistics"""
        total_tests = len(self.results)
        avg_accuracy = sum(r['accuracy'] for r in self.results) / total_tests
        avg_time = sum(r['time_taken'] for r in self.results) / total_tests
        
        print("\n" + "="*70)
        print("📊 SUMMARY STATISTICS")
        print("="*70)
        print(f"Total Tests: {total_tests}")
        print(f"Average Accuracy: {round(avg_accuracy, 2)}%")
        print(f"Average Time: {round(avg_time, 2)} seconds")
        
        print("\n" + "="*70)
        print("📋 DETAILED RESULTS")
        print("="*70)
        print(f"{'Document Type':<25} | {'Accuracy':<10} | {'Time':<10} | {'Status':<8}")
        print("-"*70)
        for r in self.results:
            print(f"{r['document_type']:<25} | {r['accuracy']:>6.2f}%   | {r['time_taken']:>6.2f}s  | {r['status']}")
        print("="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Define test cases - UPDATE WITH YOUR IMAGE PATHS AND EXPECTED TEXT
    test_cases = [
        {
            'image_path': 'test_images/english_printed.png',
            'document_type': 'English Printed',
            'expected_text': 'Today was a wonderful day. I went for a long walk in the park in the morning. The weather was warm and sunny, and the park was serene. In the afternoon, I played football with some friends. We had a lot of fun and even won the game! I\'m feeling grateful for such a great day.'
        },
        {
            'image_path': 'test_images/english_handwritten.png',
            'document_type': 'English Handwritten',
            'expected_text': 'Once there was a dog.He was very hungry.He found a piece of meat.He reaches on a bank of river.He saw his shadow in the water.He thought there was another dog in water. He became greedy. He barked to shadow to get meat.His piece of meat fell down in water.He lost his piece of meat.'
        },
        {
            'image_path': 'test_images/nepali_printed.png',
            'document_type': 'Nepali Printed',
            'expected_text': 'अर्थ हुन् खुशी कि सम्बन्ध हुन् खुशी स्वतन्त्र हुन् खुशी या परिवारसँग हुन् खुशी बाहुङ्गा खुशी कि आजै गयुँगा खुशी'
        },
        {
            'image_path': 'test_images/nepali_handwritten.png',
            'document_type': 'Nepali Handwritten',
            'expected_text': 'होस्पियारीका चेतावनी सबैलाई दिदै नानीथङ्कुलाई घरमै छोडेर दिनभरि ख्येतमा काम गर्न जाने अनुकुल मिलारुका शिव नारान रक्दै त्यतिबेला ख्येतातिर दौडे । रात–भरिकी इत्लिइउड अनिद्राप्रा शिव नारान रक्दै त्यतै बेदै नानीथङ्कुका बारीमा निवर्की चित्ला गरे, नाना सोच विचार गरे । अब नानीथङ्कुलाई झट्ट रुष्टा सुपात्रास्सित् झिङै गरेर नदिइ यसै राती छोडनु उनले ठीक ठानेनन् । चरैंखे आफ्नै नानीथङ्कुलाई रुष्टा सुधोग्य घर चोजो गरेर पनि दुई चार दिनैमै विवाह गरौं निश्चन्त हुन् पर्यो ।'
        },
    ]
    
    # Create evaluator
    evaluator = DocumentEvaluator()
    
    # Run evaluation
    evaluator.run_evaluation(test_cases)
    
    # Print summary
    evaluator.print_summary()
    
    print("\n✨ Evaluation complete!")
