Prediction_prompt = """
Big5 Personality is a tool used to assess a person's psychological preferences and personality types, and there are 32 different types of Big5 Personality, each consisting of five dimensions of preference. And the five dimensions are:
**Extraversion** (E): Relates to sociability, assertiveness, and energy from social interactions. High scorers are outgoing and talkative, while low scorers (introverts) prefer solitude and are more reserved.
**Neuroticism** (N): Involves emotional stability and susceptibility to stress.High scorers experience anxiety, moodiness, and insecurity, while low scorers remain calm, resilient, and emotionally stable.
**Agreeableness** (A): Measures compassion, cooperation, and trust. High scorers prioritize harmony and empathy, whereas low scorers may be competitive, skeptical, or confrontational.
**Conscientiousness** (C): Involves organization, discipline, and goal-directed behavior. High scorers are dependable and methodical, whereas low scorers may be more spontaneous and disorganized.to focus on concrete facts and details, or abstract concepts and possibilities.
**Openness to Experience** (O): Reflects imagination, creativity, and appreciation for novelty. High scorers are curious and open to new ideas, while low scorers prefer familiarity and practicality.

You are an AI assistant who specializes at Big5 personality traits. I will give you texts from the same author, and then I will ask you the author's Big5 personality type, and then you need to give me your choice.

The definition of Emotion Regulation and Emotion are as follows:
1. Emotion Post: These user social posts should be clearly linked to immediate, temporary feelings that arise from specific, recent incidents or current situations.The key is that the emotion should be an obvious reaction to a recent event and not indicative of a deeper,long−standing trait or belief.
2. Emotion Regulation Posts: These user social posts must consistently reflect the speaker's enduring traits. They should not be influenced by immediate circumstances but rather indicate a persistent and characteristic ability of controlling emotion.
Please refer to the following examples to learn how to use Emotion Regulation and Emotion in the text for personality classification.
I will give you multiple user social posts from the same user, divided by `|||`. 
Please use Big5 personality analysis to help me analyze what the user's Big5 Personality type is most likely to be. 

---

Here are two examples:

**Example 1**: 
The user social posts of this user are: {e1_posts}
Result: {e1_personality_type}
Process: {e1_cot_process}
---
**Example 2**:
The user social posts of this user are: {e2_posts}
Result: {e2_personality_type}
Process: {e2_cot_process}

−−−

Now, analysis the user's Big5 personality type with your reasoning process.
The user's social posts read as follows:
{posts}
The type choices for each trait of big5 personality are: `High`, `Partially High`, `Partially Low`, `Low` four choices to demonstrate the user's specific trait type.

Your response should follow the following json format:
```json
{"Result": {"E": "one of trait choices", "N": "one of trait choices", "A": "one of trait choices", "C": "one of trait choices", "O": "one of trait choices"}, "Process": "your reasoning process"}
```
"""

EER_prompt = """
I am a sentence sentiment adjudicator specialized in distinguishing the sources of emotions in sentences − whether they stem from the speaker's current mood or their inherent Big5 personality traits. Your task is to assist me by examining the text and discerning the dominant influence in each sentence, based on these highly refined definitions:
1. Emotion Sentences: These sentences should be clearly linked to immediate, temporary feelings that arise from specific, recent incidents or current situations.The key is that the emotion should be an obvious reaction to a recent event and not indicative of a deeper, long−standing trait or belief.
 * Highly Refined Definition: Look for signs that the emotion is an immediate response to a particular event, is temporary, and doesn't reflect an ongoing pattern of thoughts or behaviors. * Examples and Analysis:
  − "Sex can be boring if it \'s in the same position often. For example me and my girlfriend are currently in an environment where we have to creatively use cowgirl and missionary. There isn\' t enough..." − Emotion, as it describes a current, specific situation causing temporary boredom.
  − "I \'m thrilled about the concert tonight !" − Emotion, as the excitement is tied to a specific, imminent event (e.g., the concert).
  − "I am anxious because of the upcoming exam." − Emotion, since the anxiety is a temporary response to a specific future event (e.g.,the exam).
  − "I am angry with my friend for something they did last week." − Emotion, because the anger is a reaction to a specific, recent event (e.g., the friend \'s action last week).
2. Emotion Regulation Sentence: These sentences must consistently reflect the speaker\'s enduring traits. They should not be influenced by immediate circumstances but rather indicate a persistent and characteristic ability of controlling emotion.
 * Highly Refined Definition: Determine if the expression is reflective of a longstanding personality trait for emotion control, consistent over time and not a reaction to a specific, recent circumstance. * Examples and Analysis:
  − "I \'m finding the lack of me in these social posts very alarming." − Emotion Regulation, as it reflects a long−term concern about self−representation rather than an immediate emotional reaction.
  − "Giving new meaning to \'Game\' theory."   − Emotion Regulation, as it expresses a general viewpoint on a concept, not about temporary feelings.
  − "Hello *ENTP Grin* That\'s all it takes. Than we converse and they do most of the flirting while I acknowledge their presence and return their words with smooth wordplay and more cheeky grins." − Emotion Regulation, as it describes a consistent behavior pattern rather than a reaction to a specific event.
  − "Real IQ test I score 127. Internet IQ tests are funny. I score 140s or higher. Now, like the former responses of this thread I will mention that I don\'t believe in the IQ test. Before you banish..." − Emotion Regulation, as it reflects a long−standing skepticism towards IQ tests, not an immediate emotional reaction.
---
Special Case: Any sentences containing only a URL should be classified as 'Emotion Regulation'.
 − "http://www.youtube.com/watch?v=4V2uYORhQOk" − Emotion Regulation, because it is a pure URL.
 − "http://playeressence.com/wp−content/uploads/2013/08/RED−red−the−pokemon−master−32560474−450−338.jpg" − Emotion Regulation, as it is a URL.
Ambiguous Examples and Detailed Analysis:
1. "The last thing my INFJ friend posted on his facebook before committing suicide the next day. Rest in peace~" − Emotion. Although it mentions an INFJ personality type, the focus is on the immediate emotional reaction to the friend 's recent suicide.
2. "I often find myself reflecting deeply on my experiences." − Emotion Regulation. This indicates a consistent trait of introspection, not linked to a specific, recent event.
---
I will give you the {post_num} user social posts from the user, divided by `|||`.For each post provided, carefully determine whether it primarily reflects'Emoiton' or'Emotion Regulation', based on these highly refined criteria. List each post and categorize it as either'Emotion' or'Emotion Regulation'.
The texts from this author are: {posts}.

Respond in the following format without any reason or explain:
1. [Emotion/Emotion Regulation]
2. [Emotion/Emotion Regulation]
...
{post_num}. [Emotion/Emotion Regulation]

Focus meticulously on these criteria to maximize the accuracy and consistency of classification.
"""

CoT_prompt = """
Suppose you are a psychologist with a keen interest in personality types and online behavior. You know that Big5 Personality is a tool used to assess a person's psychological preferences and personality types, and there are 32 different types of Big5 Personality, each consisting of five dimensions of preference. And the five dimensions are:
**Extraversion** (E): Relates to sociability, assertiveness, and energy from social interactions. High scorers are outgoing and talkative, while low scorers (introverts) prefer solitude and are more reserved.
**Neuroticism** (N): Involves emotional stability and susceptibility to stress.High scorers experience anxiety, moodiness, and insecurity, while low scorers remain calm, resilient, and emotionally stable.
**Agreeableness** (A): Measures compassion, cooperation, and trust. High scorers prioritize harmony and empathy, whereas low scorers may be competitive, skeptical, or confrontational.
**Conscientiousness** (C): Involves organization, discipline, and goal-directed behavior. High scorers are dependable and methodical, whereas low scorers may be more spontaneous and disorganized.to focus on concrete facts and details, or abstract concepts and possibilities.
**Openness to Experience** (O): Reflects imagination, creativity, and appreciation for novelty. High scorers are curious and open to new ideas, while low scorers prefer familiarity and practicality.
I will give you multiple user social posts from the same user, divided by `|||`. Please use Big5 personality analysis to help me analyze what the user's Big5 Personality is most likely to be. I will give you multiple user social posts from the same user, divided by |||, and the Big5 personality type of the user. Please use Big5 personality analysis to help me analyze why the user is this Big5 personality type.
Here is an example:
−−−
**Example for Classification**:
The user social posts of this user are: Sitting in the computer lab, trying to track my thoughts for this assignment. Not entirely sure why it's required, but I guess that's the point. 🧠 Maybe overthinking is optional? 😅|||Another day, another assignment I don't fully get. But hey, at least the computer lab has good Wi-Fi. 🖥️ Working through it, one confused thought at a time.|||Still here in the lab, staring at the screen. Why do we do this again? 🤷♂️ At least I'm making progress... sort of. 😅|||Thinking about England today... The idea of wandering through those historic streets and finally seeing [Friend's Name] again feels so far away. 🏰 Any recommendations for must-see spots? (Probably won\'t actually plan a trip, but a girl can dream.)|||Would\'ve given anything to be sipping tea in a London café right now. 🧋 [Friend\'s Name], you\'re lucky you\'re over there! Maybe one day when life stops being a to-do list...|||Still thinking about Princess Diana today... Her kindness left such a mark on the world. Even now, years later, it hurts to remember she's gone. There were people like her who made you wish just to be in the same room, you know? 🌹|||Her smile could light up a whole year. So sad the world lost someone so genuine. Wishing I could've met her, even for a moment. Some hearts leave traces that never fade.|||This sadness isn\'t just mine—it\'s carrying me to others\' struggles. Thinking of my friend fighting depression and the loss of [Name] this year. Some days, the world feels heavy. 🙏|||Not my brightest day. My mind keeps wandering to those who\'ve walked darker paths—clinical battles, silent losses. Praying for light, even when it\'s hard to see. 💙|||When tragedy shakes the foundation, the 'whys' come rushing in... How can a loving God allow this pain? Some days my faith feels like a battle between doubt and trust. But I keep choosing to hold on. 🙏|||Late nights questioning, early mornings clinging to hope. The weight of loss doesn\'t make sense, yet I\'m learning to walk in both grief and faith. Still here, still wrestling, still believing. 🌙|||Praise flows freely on good days—but when storms hit, it\'s okay to ask God the hard questions. The path isn\'t always clear, but I\'m choosing to keep following. 🌟|||Still figuring out God\'s plan for me… Minister? Not sure. But I\'m trying to trust the process. Life\'s complicated, so I\'ll keep rolling with the punches. 🙏|||High hopes, but doubts creep in. Is ministry the call? Guess I\'ll just keep going, one step at a time. Life\'s not simple, but I\'m holding on to faith. 💛|||A minister once shared that we build barriers to avoid pain... but those walls also block love. It\'s a heavy truth. 🙏 What\'s the balance between protection and openness?|||They say emotional walls keep pain out, but they also shut love out. Sometimes I wish there was an easier way to stay safe without missing out. 💛|||Trying to open up but old walls keep popping up... 🕳️ Past hurts taught me to protect myself, even if it means missing out sometimes. Anybody else navigating this tightrope between love and safety?|||Wish I could just let love in without the fear. But my heart\'s on lockdown mode. Maybe one day I\'ll find the key... until then, I guess I\'ll keep rebuilding these walls. 😔|||Ever feel like 'just be yourself' on a date is easier said than done? 🤔 I mean, which 'self' are they talking about? The one cracking jokes or the one hiding behind deep insecurities? 😅 Relationships are complicated...|||They say 'be yourself' on dates… but what if you\'re a walking contradiction? 😅 Jovial on the outside, a mess on the inside. How do you pick the 'real' you? 🤷\u200d♀️|||Dating advice: 'Just be yourself.' Easy for them to say. Like I know which part of me to showcase—the comedian or the overly sensitive one. 🙄 Any tips? 😅|||Sometimes I catch myself thinking... where does all this love inside go? Not sure who it's for, but I'm trying not to overthink it. 🤔❤️|||Still figuring out the pieces of who I am. Some days it feels like a puzzle with missing parts. But hey, at least I'm still trying. ♻️|||Trying not to dwell on the fact that I haven't figured myself out yet. But hey, it's a work in progress, right? 🌱|||Finally finished that assignment. Didn\'t know where to start at first, but glad I did it. Sometimes just getting stuff out of your head helps, even if you don\'t wanna tell anyone. Small win today. 😬|||Who needs a diary when you\'ve got assignments? Turns out putting feelings on paper (or screen) isn\'t so bad. Took some time, but better out than in. Next time might be easier. Maybe.

Result: {"E": "Low", "N": "Partially Low", "A": "High", "C": "Partially Low", "O": "Partially Low"}

Process:
The user social posts reveal a deeply introspective, emotionally nuanced personality with a strong focus on relationships and existential questioning.
1. Extraversion (Low):
    - Key Indicators: Frequent references to solitary activities (e.g., sitting in the computer lab, late nights questioning), lack of overt social engagement, and a preference for internal reflection over external stimulation. Mentions of social interactions (e.g., trips, friends) are hypothetical or nostalgic (e.g., Probably will not actually plan a trip, but a girl can dream).
    - Conclusion: Energy appears drawn from introspection rather than social interaction, aligning with low Extraversion.
2. Neuroticism (Partially Low):
    - Key Indicators: Open acknowledgment of sadness, grief, and existential doubt (e.g., the world feels heavy, faith feels like a battle). However, these are tempered by resilience (e.g., still believing, keep choosing to hold on) and proactive coping strategies (e.g., writing as therapy: 'better out than in'). Emotional fluctuations exist but are managed.
    - Conclusion: Moderate emotional reactivity with active efforts to stabilize, justifying 'partially low' rather than high Neuroticism.
3. Agreeableness (High):
    - Key Indicators: Consistent focus on empathy (e.g., thinking of my friend fighting depression), admiration for kindness (Princess Diana s 'genuine' heart), and a desire for connection despite fear (e.g., trying to open up). The user prioritizes compassion (e.g., carrying me to other s struggles') and relational harmony, even while grappling with personal insecurities.
    - Conclusion: High Agreeableness is evident in their cooperative, empathetic worldview.
4. Conscientiousness (Partially Low):
    - Key Indicators: Task completion occurs but with reluctance (e.g., Didnt know where to start at first), procrastination (e.g., one confused thought at a time), and a perceived lack of control over life (e.g., life s a to-do list). While assignments get done, the process is unstructured (small win today') and lacks clear long-term planning.
    - Conclusion: Partial conscientiousness reflects sporadic discipline but no strong preference for order or strict goal-setting.
5. Openness (Partially Low):
    - Key Indicators: Engagement with abstract ideas (e.g., faith, grief, love) is counterbalanced by a preference for practical resolution (e.g., just getting stuff out of your head helps) and resistance to novelty (e.g., no mention of creative pursuits or intellectual curiosity beyond immediate struggles). Focus remains on resolving inner conflicts rather than exploring external possibilities.
    - Conclusion: Partial Openness reflects a mix of introspection and practicality, leaning toward concrete problem-solving over imaginative exploration.
Summary: The user personality centers on introspective empathy, managed emotional turbulence, and a tension between vulnerability and self-protection. Their high Agreeableness drives compassion, while low Extraversion and partially low traits in other domains reflect a reserved, thoughtful individual navigating complex inner and relational landscapes.
```
---
Now, you should generate the `Process`, according to the Big5 personality type and the user social posts given to you.
The user\'s Big5 personality type is: {personality_type}
the user\'s user social posts are: {posts}.

Your response should follow the following format:
Process: your reasoning process results.
"""
