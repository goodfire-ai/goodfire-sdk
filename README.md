# Quickstart


Ember is a hosted API/SDK that lets you shape AI model behavior by directly controlling a model's internal units of computation, or **"features"**.
With Ember, you can modify features to precisely control model outputs, or use them as building blocks for tasks like classification.

In this quickstart, you'll learn how to:

- Find features that matter for your specific needs
- Edit features to create model variants
- Discover which features are active in your data
- Save and load your model variants


```python Code
pip install goodfire
```


You can get an API key through [our platform](https://platform.goodfire.ai)


```python Code
GOODFIRE_API_KEY = "{YOUR_API_KEY}"
```


## Initialize the SDK


```python Code
import goodfire
client = goodfire.Client(api_key=GOODFIRE_API_KEY)
# Instantiate a model variant. 
variant = goodfire.Variant("meta-llama/Llama-3.3-70B-Instruct")
```


Our sampling API is OpenAI compatible, making it easy to integrate.


```python Generation Code
for token in client.chat.completions.create(
    [{"role": "user", "content": "Hi, how are you?"}],
    model=variant,
    stream=True,
    max_completion_tokens=100,
):
    print(token.choices[0].delta.content, end="")
```
```output Generation Output
I'm doing great, thanks for asking. How about you? How can I help you today?
```


# Editing features to create model variants
## How to find relevant features for edits
There are three ways to find features you may want to modify:

- **Auto Steer**: Simply describe what you want, and let the API automatically select and adjust feature weights

- **Feature Search**: Find features using semantic search

- **Contrastive Search**: Identify relevant features by comparing two different datasets

Let's explore each method in detail.

## Auto steering

Auto steering automatically finds and adjusts feature weights to achieve your desired behavior. Simply provide a short prompt describing what you want, and autosteering will:

- Find the relevant features
- Set appropriate feature weights
- Return a FeatureEdits object that you can set directly


```python Auto Steer Code
edits = client.features.AutoSteer(
    specification="be funny",  # or your desired behavior
    model=variant,
)
variant.set(edits)
print(edits)
```
```output Auto Steer Output
FeatureEdits([
   0: (Setup phrases in jokes and narratives, 0.6875)
   1: (The assistant is checking if their joke landed well, 0.41250000000000003)
])
```


Now that we have a few funny edits, let's see how the model responds!


```python Auto Steer Generation Code
for token in client.chat.completions.create(
    [{"role": "user", "content": "Tell me about pirates"}],
    model=variant,
    stream=True,
    max_completion_tokens=120,
):
    print(token.choices[0].delta.content, end="")
```
```output Auto Steer Generation Output
Pirates! They're always a treasure to talk about! Get it? Treasure? Okay, let's dive into it.
Pirates were seafarers who sailed the seven seas, pillaging and plundering ships. They were known for their bravery, sword-fighting skills, and... bad navigation skills! Arrr, I mean, they were always lost at sea!
But seriously, pirates like Blackbeard, Calico Jack, and Captain Hook were infamous for their fearlessness and cunning. They'd often target merchant ships, stealing gold, silver, and other precious booty.
Pirate
```

The model automatically added puns/jokes, even though we didn't specify anything about comedy in our prompt.

## Feature search

Let's reset the model to its default state (without any feature edits)


```python Code
variant.reset()
```


Feature search helps you explore and discover what capabilities your model has. It can be useful when you want to browse through available features.


```python Feature Search Code
funny_features = client.features.search(
    "funny",
    model=variant,
    top_k=10
)
print(funny_features)
```
```output Feature Search Output
FeatureGroup([
   0: "descriptions of sophisticated or edgy senses of humor",
   1: "The assistant is explaining why something is funny",
   2: "Explaining why something is funny or humorous",
   3: "The assistant is explaining why wordplay or puns are funny",
   4: "The assistant is explaining why a joke or pun is supposed to be funny",
   5: "Humor being used to improve situations or relationships",
   6: "Nouns that are the subject of simple jokes or puns",
   7: "Explaining or qualifying humorous/lighthearted content",
   8: "Fun/entertaining/amusing across Romance languages",
   9: "The assistant should respond in a playful or humorous tone"
])
```

When setting feature weights manually, start with 0.5 to enhance a feature and -0.3 to ablate a feature. When setting multiple features, you may need to tune down the weights.

```python Feature set and generation code
variant.set(funny_features[0], 0.6)
for token in client.chat.completions.create(
      [
          {"role": "user", "content": "tell me about foxes"}
      ],
      model=variant,
      stream=True,
      max_completion_tokens=100,
  ):
      print(token.choices[0].delta.content, end="")
```
```output Feature set and generation output
Foxes! They're quite the charmers. Here are some key facts:
* They're part of the Canidae family, related to dogs and wolves.
* There are 12 species, including the red fox, arctic fox, and fennec fox.
* Foxes are known for their sharp wit, agility, and sarcasm (just kidding about that last one, but they do have a great sense of humor... or so it seems).
* They're omnivores, eating
```

Feel free to play around with the weights and features to see how the model responds.


#### (Advanced) Look at a feature's nearest neighbors


Get neighboring features by comparing them to either individual features or groups of features. When comparing to individual features, `neighbors()` looks at similarity in the embedding space. When comparing to groups, `neighbors()` finds features closest to the group's centroid.

`neighbors()` helps you understand feature relationships beyond just their labels. It can reveal which features might work best for your intended model adjustments.


```python Nearest Neighbors Code
client.features.neighbors(
    funny_features[0],
    model=variant,
    top_k=5
)
```
```output Nearest Neighbors Output
FeatureGroup([
   0: "Explaining or qualifying humorous/lighthearted content",
   1: "Detecting and responding appropriately to requests for potentially inappropriate humor",
   2: "The assistant is providing multiple jokes or anecdotes",
   3: "Sarcastic or cynical tone detection",
   4: "The assistant should reject offensive or harmful content requests"
])
```

Now, you can find more features that are similar to other features


```python Nearest Neighbors
client.features.neighbors(
    funny_features[2],
    model=variant,
    top_k=5
)
```
```output Out
FeatureGroup([
   0: "Punctuation marking the punchline delivery in jokes",
   1: "The assistant is providing a numbered list of options or examples",
   2: "The user is requesting or discussing jokes",
   3: "The assistant is about to present a joke or example",
   4: "The assistant's turn to begin speaking in the conversation"
])
```

## Contrastive search

Contrastive search lets you discover relevant features in a data-driven way.

Provide two datasets of chat examples:

- dataset_1: Examples of behavior you want to avoid
- dataset_2: Examples of behavior you want to encourage

Examples are paired such that the first example in dataset_1 contrasts the first example in dataset_2, and so on.


#### Reranking
Contrastive search becomes more powerful when combined with reranking. First, contrastive search finds features that distinguish between your datasets. Then, reranking sorts these features using your description of the desired behavior.

This two-step process ensures you get features that are both:

- Mechanistically useful (from contrastive search)
- Aligned with your goals (from reranking)


Let's specify two conversation datasets. The first has a typical helpful assistant response and the second assistant replies in jokes.


```python Contrastive Search Code
variant.reset()
default_conversation = [
    [
        {
            "role": "user",
            "content": "Hello how are you?"
        },
        {
            "role": "assistant",
            "content": "I am a helpful assistant. How can I help you?"
        }
    ]
]
joke_conversation = [
    [
        {
            "role": "user",
            "content": "Hello how are you?"
        },
        {
            "role": "assistant",
            "content": "What do you call an alligator in a vest? An investigator!"
        }
    ]
]
helpful_assistant_features, joke_features = client.features.contrast(
    dataset_1=default_conversation,
    dataset_2=joke_conversation,
    model=variant,
    top_k=30
)
# Let's rerank to surface humor related features
joke_features = client.features.rerank(
    features=joke_features,
    query="funny",
    model=variant,
    top_k=5
)
joke_features
```
```output Contrastive Search Output
FeatureGroup([
   0: "Punchline moments in setup-punchline format jokes",
   1: "Setup phrases for question-answer format jokes",
   2: "Transition between joke setup and punchline",
   3: "The assistant is checking if their joke landed well",
   4: "Setup phrases in jokes and narratives"
])
```


We now have a list of features to consider adding. Let's set some plausible-looking ones from `joke_features`.

Note that we could also explore removing some of the helpful_assistant features.

```python Contrastive Search Set Features Code
variant.reset()
variant.set(joke_features[0,1], 0.6)
for token in client.chat.completions.create(
    [
        {"role": "user", "content": "Hello. Tell me about whales."}
    ],
    model=variant,
    stream=True,
    max_completion_tokens=100,
):
    print(token.choices[0].delta.content, end="")
```
```output Contrastive Search Set Features Output
You call a group of whales? A pod? Because you have a whale of a time when you're having a party? I said to the whale? "Why did you go to the party?" He said, "Because I heard it was an orca-stra?" I'm fin-tastic at telling jokes about the ocean. I'm reading a book about a fish? It's fin-tastic? I'm fin-ished? I'm reading a book about a fish? It's fin
```

## (Advanced) Conditional logic for feature edits


You can establish relationships between different features (or feature groups) using conditional interventions.


First, let's reset the variant and pick out the funny features.


```python Conditional Logic Code
variant.reset()
funny_features
```
```output Conditional Logic Output

FeatureGroup([
   0: "descriptions of sophisticated or edgy senses of humor",
   1: "The assistant is explaining why something is funny",
   2: "Explaining why something is funny or humorous",
   3: "The assistant is explaining why wordplay or puns are funny",
   4: "The assistant is explaining why a joke or pun is supposed to be funny",
   5: "Humor being used to improve situations or relationships",
   6: "Nouns that are the subject of simple jokes or puns",
   7: "Explaining or qualifying humorous/lighthearted content",
   8: "Fun/entertaining/amusing across Romance languages",
   9: "The assistant should respond in a playful or humorous tone"
])
```

Now, let's find a features where the model is talking like a pirate.


```python Pirate Features Code
pirate_features = client.features.search(
    "talk like a pirate",
    model=variant,
    top_k=1
)
print(pirate_features[0])
```
```output Pirate Features Output
Feature("The assistant should roleplay as a pirate")
```

Now, let's set up behaviour so that when the model is talking like a pirate, it will be funny.

```python Conditional Logic Set Features Code
variant.set_when(pirate_features[0] > 0.75, {
    funny_features[0]: 0.7,
})
# The model will now try to be funny when talking about pirates
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "talk like a pirate and tell me about whales"}],
    model=variant
)
print(response.choices[0].message["content"])
```
```output Conditional Logic Set Features Output
Yer lookin' fer some oceanic humor, eh? Alright, let's set sail fer some laughs. I've got a few sea stories, but I'll keep 'em as dry as the sense of humor of a retired pirate. Here's one: Why did the sea captain bring a sense of humor? Because he's always funny, even on land. That's how I find humor. Now, let's move on to the sense of humor. Why was it funny? Well, it's not, but I'm here to tell you that's what I call a sense of humor. Now, let's talk about everything else, but not funny. I'm done here.
```


Say we decide the model isn't very good at pirate jokes. Let's set up behavior to stop generation altogether if the pirate features are too strong.


```python Abort when Pirate Features are too strong
# Abort if pirate features are too strong
variant.abort_when(pirate_features > 0.75)
try:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Tell me about pirates."}],
        model=variant
    )
except goodfire.exceptions.InferenceAbortedException:
    print("Generation aborted due to too much pirate content")
```


If you aren't sure of the features you want to condition on, use AutoConditional with a specified prompt to get back an automatically generated condition.


```python Auto Conditional Code
# Generate auto conditional based on a description. This will automatically
# choose the relevant features and conditional weight
conditional = goodfire.conditional.AutoConditional(
    "pirates",
    client=client,
    model="meta-llama/Llama-3.3-70B-Instruct",
    num_features_to_use=5
)

# Apply feature edits when condition is met
variant.set_when(conditional, {
    joke_features[0]: 0.5,
    joke_features[1]: 0.5
})
```


# Discover which features are active in your data


### Working with a conversation context


You can inspect what features are activating in a given conversation with the `inspect` API, which returns a `context` object.


Say you want to understand what model features are important when the model tells a joke. You can pass in the same joke conversation dataset to the inspect endpoint.

```python Inspect Code
variant.reset()
context = client.features.inspect(
    messages=joke_conversation[0],
    model=variant,
)
context
```
```output Inspect Output
ContextInspector(
   <|begin_of_text|><|start_header_id|>system<|end_header_id|>
   
   Cutting Knowledge Date: December 2023
   Today Date: 26 Jul 2024
   
   <|eot_id|><|start_header_id|>user<|end_header_id|>
   
   Hello how are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
   
   What do you call an alligator in a vest...
)
```

From the context object, you can access a lookup object which can be used to look at the set of feature labels in the context.

```python Lookup Code
lookup = context.lookup()
lookup
```
```output Lookup Output
{41637: Feature("Start of a new conversation segment in chat format"),
 22058: Feature("Start of a new conversation segment"),
 13884: Feature("Start of a new conversation segment in chat format"),
 38729: Feature("Start of a new conversation or major topic reset"),
 ...
 42053: Feature("The assistant is providing a list of options for how to respond to a social situation")}
```

You can select the top `k` activating features in the context, ranked by activation strength. There are features related to jokes and tongue twisters, among other syntactical features.

```python Inspect Top Features Code
top_features = context.top(k=10)
top_features
```
```output Inspect Top Features Output
FeatureActivations(
   0: (Feature("Assistant's initial friendly greeting to open a conversation"), 9)
   1: (Feature("Setup phrases in jokes and narratives"), 7)
   2: (Feature("Formal definition or description of a concept"), 6)
   3: (Feature("Reptiles and reptilian characteristics"), 6)
   4: (Feature("Grammatical patterns describing training, transformation, or movement processes"), 6)
   5: (Feature("Informal conversation-starting greetings from users across languages"), 5)
   6: (Feature("The assistant needs clarification or is offering help"), 5)
   7: (Feature("Setup phrases for question-answer format jokes"), 5)
   8: (Feature("Detailed anatomical descriptions of semi-aquatic animals"), 5)
   9: (Feature("Comma-separated items in structured data lists, especially quoted strings"), 4)
)
```


You can also inspect individual tokens level feature activations. Let's see what features are active at the punchline token.


```python Token Activations Code
print(context.tokens[-3])
token_acts = context.tokens[-3].inspect()
token_acts
```
```output Token Activations Output
FeatureActivations(
   0: (Feature("Active investigation or investigative activities"), 2.515625)
   1: (Feature("Detective fiction and investigative narratives"), 1.1796875)
   2: (Feature("Answer key terms in educational testing contexts"), 1.140625)
   3: (Feature("Complaints about repetitive or stale situations"), 1.0)
   4: (Feature("Flirtatious or suggestive roleplay interactions"), 0.98046875)
)
```


#### (Advanced) Look at next token logits


```python Next Logits Code
logits = client.chat.logits(
    messages=joke_conversation[0],
    model=variant,
)
logits.logits
```
```output Next Logits Output
{' I': 18.75,
 ' Would': 17.125,
 ' Hope': 16.625,
 ' How': 16.5,
 '<|eot_id|>': 15.5625,
 ...
 ' Relax': 4.84375,
 'opening': 4.84375,
 ...}
```


### Get feature activation vectors for machine learning tasks


To run a machine learning pipeline at the feature level (for instance, for humor detection) you can directly export features using `client.features.activations` to get a matrix or retrieve a sparse vector for a specific `FeatureGroup`.


```python Matrix Code
activations = client.features.activations(
    messages=joke_conversation[0],
    model=variant,
)
activations
```
```output Matrix Output
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])
```


```python Vector Code
top_features.vector()
```
```output Vector Output 
array([0., 0., 0., ..., 0., 0., 0.])
```


## Inspecting specific features


There may be specific features whose activation patterns you're interested in exploring. In this case, you can specify features such as *humor_features* and pass that into the `features` argument of `inspect`.


```python Humor Features Code
humor_features = client.features.search("jokes and humor", model=variant, top_k=15)
humor_features
```
```output Humor Features Output
FeatureGroup([
   0: "The conversation involves telling or requesting jokes",
   1: "Meta-discussions and classifications of humor",
   2: "Humor being used to improve situations or relationships",
   3: "The assistant is providing multiple jokes or anecdotes",
   4: "Comedy and comedians across languages",
   5: "Setup phrases for question-answer format jokes",
   6: "Nouns that are the subject of simple jokes or puns",
   7: "descriptions of sophisticated or edgy senses of humor",
   8: "Setup portion of question-answer format jokes, especially 'Why did the...'",
   ...
   14: "Witty banter and clever dialogue exchanges"
])
```


Now, let's see if these features are activating in the joke conversation.


```python Inspect Humor Features Code
context = client.features.inspect(
    messages=joke_conversation[0],
    model=variant,
    features=humor_features
)
context
```
```output Inspect Humor Features Output
ContextInspector(
   <|begin_of_text|><|start_header_id|>system<|end_header_id|>
   
   Cutting Knowledge Date: December 2023
   Today Date: 26 Jul 2024
   
   <|eot_id|><|start_header_id|>user<|end_header_id|>
   
   Hello how are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
   
   What do you call an alligator in a vest...
)
```

Now you can retrieve the top k activating *humor features* in the `context`. This might be a more interesting set of features for downstream tasks.

```python Top Features Code
humor_feature_acts = context.top(k=5)
humor_feature_acts
```
```output Top Features Output
FeatureActivations(
   0: (Feature("Setup phrases in jokes and narratives"), 9)
   1: (Feature("Setup phrases for question-answer format jokes"), 7)
   2: (Feature("Transition between joke setup and punchline"), 3)
   3: (Feature("The assistant is about to tell jokes"), 2)
   4: (Feature("Nouns that are the subject of simple jokes or puns"), 2)
)
```


# Save and load your model variants


You can serialize a variant to JSON format for saving.


```python Save Variant Code
variant.reset()
variant.set(pirate_features[0], 0.9)
variant_json = variant.json()
variant_json
```
```output Save Variant Output
{'base_model': 'meta-llama/Llama-3.3-70B-Instruct',
 'edits': [{'feature_id': 'a16920eb-459c-4862-9c42-d0711c9c700c',
   'feature_label': 'The assistant should roleplay as a pirate',
   'index_in_sae': 33234,
   'value': 0.9}],
 'scopes': []}
```


And load a variant from JSON format.


```python Load Variant Code
loaded_variant = goodfire.Variant.from_json(variant_json)
loaded_variant
```
```output Load Variant Output
Variant(
   base_model=meta-llama/Llama-3.3-70B-Instruct,
   edits={
      Feature("The assistant should roleplay as a pirate"): 0.9,
   }
)
```


Now, let's generate a response with the loaded variant.


```python Generate Response Code
for token in client.chat.completions.create(
    [
        {"role": "user", "content": "tell me about whales"}
    ],
    model=loaded_variant,
    stream=True,
    max_completion_tokens=150,
):
    print(token.choices[0].delta.content, end="")
```
```output Generate Response Output
Whales! They're the largest animals on Earth, with some species reachin' lengths of over 100 feet! They're mammals, just like us, but they live in the water. There's two main types: toothed whales (like orcas) and baleen whales (like blue whales). They're super social, too, often swimmin' in big groups. Some whales can even sing songs, like humpbacks! What'd you like to know more about, friend?
```


# Using OpenAI SDK


You can also work directly with the OpenAI SDK for inference since our endpoint is fully compatible.


```python Install OpenAI Code
!pip install openai --quiet
```


```python OpenAI SDK Code
from openai import OpenAI

oai_client = OpenAI(
    api_key=GOODFIRE_API_KEY,
    base_url="https://api.goodfire.ai/api/inference/v1",
)

response = oai_client.chat.completions.create(
    messages=[
        {"role": "user", "content": "who is this"},
    ],
    model=variant.base_model,
    extra_body={"controller": variant.controller.json()},
)

response.choices[0].message.content
```

# Note

For more advanced usage and detailed API reference, check out our [SDK reference](https://docs.goodfire.ai/sdk-reference/) and [example notebooks](https://github.com/goodfire/goodfire-examples).
