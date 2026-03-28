# Declarative Goal

Provide the closest results possible in a human-in-the-loop flow capturing
dish names, ingredients, and weights of all food present in a photo, to allow
a user to accurately track their macro and micro nutrients throughout the day
based on photos of their food.

## Key Constraints

- **HITL asymmetry**: The user can delete incorrect ingredients but may not notice
  missing ones. False negatives are costlier than false positives.
- **Caloric density priority**: Errors on calorically dense ingredients (proteins,
  fats, starches) cause larger tracking inaccuracies than errors on low-calorie
  items (herbs, spices, leafy garnishes). Getting 250g of rice wrong costs ~325
  calories of error; getting 5g of sesame oil wrong costs ~45 calories.
- **Output token limit**: The target model (Gemini Nano) has a 256-token output
  limit. Multi-pass strategies are acceptable if they improve accuracy.
- **Latency**: Matters for UX but is secondary to accuracy. Under 10s total is
  acceptable; under 5s is ideal.
- **Structured output**: Results must be valid JSON. 100% parse rate is a hard
  requirement — unparseable output is a total failure.
