from api_client import FactCheckAPI

def main():
    api = FactCheckAPI()
    print("\n" + "="*45)
    print("  AI SYSTEMS: MISINFORMATION DETECTOR DEMO")
    print("="*45)
    
    claim = input("\nEnter a claim to verify (e.g., 'The moon is cheese'): ")
    
    if not claim.strip():
        print("Empty input. Please try again.")
        return

    print("Checking fact-check databases...")
    data = api.search_claim(claim)
    
    if "error" in data:
        print(f"Error: {data['error']}")
    elif "claims" in data:
        print(f"\nFound {len(data['claims'])} matching fact-check(s):")
        for i, item in enumerate(data['claims'][:3]):  # Limit to top 3 for the demo
            review = item['claimReview'][0]
            print(f"\n--- Result {i+1} ---")
            print(f"Claim: {item['text']}")
            print(f"Rating: {review['textualRating']}")
            print(f"Publisher: {review['publisher']['name']}")
            print(f"Link: {review['url']}")
    else:
        print("\nNo results found. This statement hasn't been formally fact-checked yet.")

if __name__ == "__main__":
    main()
