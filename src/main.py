import os
from api_client import FactCheckAPI
from utils.analyzer import simple_bias_check

def main():
    api = FactCheckAPI()
    print("\n" + "="*45)
    print("  AI SYSTEMS: ADVANCED MISINFO DETECTOR")
    print("="*45)
    
    claim = input("\nEnter a claim to verify: ")
    if not claim.strip(): return

    # Phase 1: Local Linguistic Analysis (Innovation Layer)
    bias_score, triggers = simple_bias_check(claim)
    
    # Phase 2: External Fact-Check Search (Data Layer)
    print("Cross-referencing with global databases...")
    data = api.search_claim(claim)
    
    # Phase 3: Decision Logic
    final_score = bias_score
    api_found = False
    rating_text = "N/A"

    if "claims" in data and len(data['claims']) > 0:
        api_found = True
        rating_text = data['claims'][0]['claimReview'][0]['textualRating']
        # If a verified fact-check exists and is negative, drop score to 0-10
        if any(neg in rating_text.lower() for neg in ['false', 'fake', 'mixture', 'pants on fire']):
            final_score = 10

    # Display Results
    print("\n" + "-"*20 + " REPORT " + "-"*20)
    print(f"Claim: {claim}")
    print(f"Fact-Check Match: {'FOUND' if api_found else 'NOT FOUND'}")
    if api_found:
        print(f"Official Rating: {rating_text}")
    
    print(f"\nFINAL RELIABILITY SCORE: {final_score}/100")
    if triggers:
        print(f"WARNING: Sensationalist language detected: {', '.join(triggers)}")
    print("-"*48)

if __name__ == "__main__":
    main()
