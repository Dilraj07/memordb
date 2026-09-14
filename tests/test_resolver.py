import pytest
from memor.extractor import Fact
from memor.resolver import resolve

CASES = [
    {
        "name": "1. Exact Duplicate (no LLM needed)",
        "existing": [{"predicate": "lives_in", "object": "Pune", "subject": "user", "id": 1}],
        "new_fact": Fact(subject="user", predicate="lives_in", object="Pune"),
        "expected": "DUPLICATE"
    },
    {
        "name": "2. Fast-Path Update (lives_in is SINGLE_VALUED, no LLM needed)",
        "existing": [{"predicate": "lives_in", "object": "Pune", "subject": "user", "id": 2}],
        "new_fact": Fact(subject="user", predicate="lives_in", object="Bangalore"),
        "expected": "UPDATE"
    },
    {
        "name": "3. Slow-Path LLM Add (additive hobbies)",
        "existing": [{"predicate": "likes", "object": "hiking", "subject": "user", "id": 3}],
        "new_fact": Fact(subject="user", predicate="likes", object="reading"),
        "expected": "ADD"
    },
    {
        "name": "4. Slow-Path LLM Update (contradictory states - e.g. vegan vs non-vegan diet)",
        "existing": [{"predicate": "diet", "object": "vegan", "subject": "user", "id": 4}],
        "new_fact": Fact(subject="user", predicate="diet", object="carnivore"),
        "expected": "UPDATE"
    },
    {
        "name": "5. Different Predicate entirely",
        "existing": [{"predicate": "lives_in", "object": "Pune", "subject": "user", "id": 5}],
        "new_fact": Fact(subject="user", predicate="works_at", object="OpenAI"),
        "expected": "ADD"
    },
    {
        "name": "6. Fast-Path Update (works_at is SINGLE_VALUED)",
        "existing": [{"predicate": "works_at", "object": "Startup", "subject": "user", "id": 6}],
        "new_fact": Fact(subject="user", predicate="works_at", object="BigTech"),
        "expected": "UPDATE"
    },
    {
        "name": "7. Duplicate with different casing",
        "existing": [{"predicate": "likes", "object": "Coffee", "subject": "user", "id": 7}],
        "new_fact": Fact(subject="user", predicate="likes", object="coffee"),
        "expected": "DUPLICATE"
    },
    {
        "name": "8. Slow-Path LLM Add (multiple pets)",
        "existing": [{"predicate": "owns_pet", "object": "dog", "subject": "user", "id": 8}],
        "new_fact": Fact(subject="user", predicate="owns_pet", object="cat"),
        "expected": "ADD"
    },
    {
        "name": "9. Slow-Path LLM Update (marital status)",
        "existing": [{"predicate": "marital_status", "object": "single", "subject": "user", "id": 9}],
        "new_fact": Fact(subject="user", predicate="marital_status", object="married"),
        "expected": "UPDATE"
    },
    {
        "name": "10. Slow-Path LLM Add (different people in context)",
        "existing": [{"predicate": "friend_with", "object": "Alice", "subject": "user", "id": 10}],
        "new_fact": Fact(subject="user", predicate="friend_with", object="Bob"),
        "expected": "ADD"
    },
    {
        "name": "11. Slow-Path LLM Update (political affiliation)",
        "existing": [{"predicate": "political_affiliation", "object": "Democrat", "subject": "user", "id": 11}],
        "new_fact": Fact(subject="user", predicate="political_affiliation", object="Republican"),
        "expected": "UPDATE"
    },
    {
        "name": "12. Slow-Path LLM Add (languages spoken)",
        "existing": [{"predicate": "speaks", "object": "English", "subject": "user", "id": 12}],
        "new_fact": Fact(subject="user", predicate="speaks", object="Spanish"),
        "expected": "ADD"
    },
    {
        "name": "13. Slow-Path LLM Update (favorite color)",
        "existing": [{"predicate": "favorite_color", "object": "blue", "subject": "user", "id": 13}],
        "new_fact": Fact(subject="user", predicate="favorite_color", object="red"),
        "expected": "UPDATE"
    },
    {
        "name": "14. Slow-Path LLM Add (allergies)",
        "existing": [{"predicate": "allergic_to", "object": "peanuts", "subject": "user", "id": 14}],
        "new_fact": Fact(subject="user", predicate="allergic_to", object="shellfish"),
        "expected": "ADD"
    },
    {
        "name": "15. Fast-Path Update (studies_at)",
        "existing": [{"predicate": "studies_at", "object": "MIT", "subject": "user", "id": 15}],
        "new_fact": Fact(subject="user", predicate="studies_at", object="Stanford"),
        "expected": "UPDATE"
    },
    {
        "name": "16. Slow-Path LLM Add (siblings)",
        "existing": [{"predicate": "has_sibling", "object": "brother named Tom", "subject": "user", "id": 16}],
        "new_fact": Fact(subject="user", predicate="has_sibling", object="sister named Sarah"),
        "expected": "ADD"
    },
    {
        "name": "17. Slow-Path LLM Add (favorite foods)",
        "existing": [{"predicate": "favorite_food", "object": "pizza", "subject": "user", "id": 17}],
        "new_fact": Fact(subject="user", predicate="favorite_food", object="sushi"),
        "expected": "ADD"
    },
    {
        "name": "18. Slow-Path LLM Update (eye color - physical trait)",
        "existing": [{"predicate": "eye_color", "object": "brown", "subject": "user", "id": 18}],
        "new_fact": Fact(subject="user", predicate="eye_color", object="blue"),
        "expected": "UPDATE"
    },
    {
        "name": "19. Slow-Path LLM Update (highest education)",
        "existing": [{"predicate": "education_level", "object": "Bachelors", "subject": "user", "id": 19}],
        "new_fact": Fact(subject="user", predicate="education_level", object="Masters"),
        "expected": "UPDATE"
    },
    {
        "name": "20. Slow-Path LLM Add (visited countries)",
        "existing": [{"predicate": "has_visited", "object": "Japan", "subject": "user", "id": 20}],
        "new_fact": Fact(subject="user", predicate="has_visited", object="Italy"),
        "expected": "ADD"
    },
    {
        "name": "21. Slow-Path LLM Update (primary phone OS)",
        "existing": [{"predicate": "phone_os", "object": "Android", "subject": "user", "id": 21}],
        "new_fact": Fact(subject="user", predicate="phone_os", object="iOS"),
        "expected": "UPDATE"
    },
    {
        "name": "22. Slow-Path LLM Add (investment portfolio)",
        "existing": [{"predicate": "invests_in", "object": "stocks", "subject": "user", "id": 22}],
        "new_fact": Fact(subject="user", predicate="invests_in", object="real estate"),
        "expected": "ADD"
    },
    {
        "name": "23. Slow-Path LLM Update (current relationship status)",
        "existing": [{"predicate": "relationship_status", "object": "dating", "subject": "user", "id": 23}],
        "new_fact": Fact(subject="user", predicate="relationship_status", object="engaged"),
        "expected": "UPDATE"
    },
    {
        "name": "24. Exact Duplicate with whitespace",
        "existing": [{"predicate": "hobby", "object": "chess", "subject": "user", "id": 24}],
        "new_fact": Fact(subject="user", predicate="hobby", object="  chess  "),
        "expected": "DUPLICATE"
    },
    {
        "name": "25. Slow-Path LLM Add (musical instruments)",
        "existing": [{"predicate": "plays_instrument", "object": "guitar", "subject": "user", "id": 25}],
        "new_fact": Fact(subject="user", predicate="plays_instrument", object="piano"),
        "expected": "ADD"
    },
    {
        "name": "26. Fast-Path Update (has_role)",
        "existing": [{"predicate": "has_role", "object": "Software Engineer", "subject": "user", "id": 26}],
        "new_fact": Fact(subject="user", predicate="has_role", object="Engineering Manager"),
        "expected": "UPDATE"
    },
    {
        "name": "27. Slow-Path LLM Add (streaming services)",
        "existing": [{"predicate": "subscribed_to", "object": "Netflix", "subject": "user", "id": 27}],
        "new_fact": Fact(subject="user", predicate="subscribed_to", object="Hulu"),
        "expected": "ADD"
    },
    {
        "name": "28. Slow-Path LLM Update (zodiac sign - immutable but mutually exclusive)",
        "existing": [{"predicate": "zodiac_sign", "object": "Aries", "subject": "user", "id": 28}],
        "new_fact": Fact(subject="user", predicate="zodiac_sign", object="Taurus"),
        "expected": "UPDATE"
    },
    {
        "name": "29. Slow-Path LLM Add (favorite movies)",
        "existing": [{"predicate": "favorite_movie", "object": "Inception", "subject": "user", "id": 29}],
        "new_fact": Fact(subject="user", predicate="favorite_movie", object="Interstellar"),
        "expected": "ADD"
    },
    {
        "name": "30. Slow-Path LLM Update (current vehicle)",
        "existing": [{"predicate": "drives", "object": "Honda Civic", "subject": "user", "id": 30}],
        "new_fact": Fact(subject="user", predicate="drives", object="Tesla Model 3"),
        "expected": "UPDATE"
    }
]

@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_resolver(case):
    decision, _ = resolve(case["new_fact"], case["existing"])
    assert decision == case["expected"]
