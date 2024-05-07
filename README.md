# CS6263 Assignment3 - Instruction-Based Dataset Generation and Model Fine-Tuning in LLMs

**Objective**: The objective of this assignment is to explore the process of generating an instruction-based dataset for model training in Natural Language Processing (NLP). Additionally, students will fine-tune a pre-trained model using the newly created instruction-based dataset and compare its performance with the original instructions. Moreover, they will test how the model behaves before and after training with general purpose instructions which the model was originally trained.

**Summary**: The first thing done was building the data set. The *jahjinx/IMDb_movie_reviews* sentiment analysis dataset was used for this assignment. Using GPT-3.5 Turbo I used the text from that dataset and made these prompts:
```
prompt = "Write a response to the review where you agree with it and another where you disagree...." + dataset[i]["text"]
prompt = prompt + "///output the response in json format with values for review, agree, and disagree"
```
The code to do this can be viewed in the `createDataset.py` file (you will need an API key though). GPT 3.5 was mostly up to the task however due to time constraints I used the `mydata.json` file from a fellow classmate. I do plan on using my own json file once it is finished being made and will put it in the `mycsvdata.csv` file.

To save on time, I decided to use the pretrained model that we made in assignment 2. After copying and renaming the model to `LlamaBase` I finetuned it further with the new imdb dataset. after which it was finetuned again with the original data from assignment 2. The 2 rounds of finetuning were done in `trainLlama2B.py` and `trainLlama2C.py`

From there, I ran my evaluation code, `Evaluation.py` and `Evaluation3B.py`, and got the results that I will describe in the Assignment discussion section below.

## Instructions
### Environment Setup
To run, first load the environment from the environment.yml file with:

`conda env create -f environment.yml`

Then activate it:

`conda activate unsloth_env`

I used the same environment as the previous assignment.

### Fine Tuning

To run the fine-tuning run:

`python trainLlama2B.py` or `python trainLlama2C.py`

### Execution

In order to run inferences for the models run:

`python Evaluation.py` then `python Evaluation3B.py`

## Assignment Discussion

**3a.) Evaluate the saved model from 2.b and 2.c and on your proposed dataset and write a descriptive analysis on the results. Create a table like the sample table provided.**

<table class="tg">
<thead>
  <tr>
    <th class="tg-9wq8" colspan="8">Normal&nbsp;&nbsp;&nbsp;Evaluation</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8"> </td>
    <td class="tg-9wq8">CodeBleu</td>
    <td class="tg-9wq8" colspan="2">Rouge-L</td>
    <td class="tg-nrix">Rouge-L Average</td>
    <td class="tg-nrix" colspan="2">BERTScore</td>
    <td class="tg-nrix">BERTScore Average</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="3">Llama2Base</td>
    <td class="tg-9wq8" rowspan="3">0.3166</td>
    <td class="tg-9wq8">Recall: </td>
    <td class="tg-nrix">0.5060</td>
    <td class="tg-nrix" rowspan="3">0.2938</td>
    <td class="tg-nrix">Recall: </td>
    <td class="tg-nrix">0.8675</td>
    <td class="tg-nrix" rowspan="3">0.8286</td>
  </tr>
  <tr>
    <td class="tg-nrix">Precision: </td>
    <td class="tg-nrix">0.1531</td>
    <td class="tg-nrix">Precision: </td>
    <td class="tg-nrix">0.7910</td>
  </tr>
  <tr>
    <td class="tg-nrix">F1 Score: </td>
    <td class="tg-nrix">0.2222</td>
    <td class="tg-nrix">F1 Score: </td>
    <td class="tg-nrix">0.8273</td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="3">Llama2 b.</td>
    <td class="tg-nrix" rowspan="3">0.1945</td>
    <td class="tg-nrix">Recall: </td>
    <td class="tg-nrix">0.4785</td>
    <td class="tg-nrix" rowspan="3">0.2621</td>
    <td class="tg-nrix">Recall: </td>
    <td class="tg-nrix">0.8467</td>
    <td class="tg-nrix" rowspan="3">0.8164</td>
  </tr>
  <tr>
    <td class="tg-nrix">Precision: </td>
    <td class="tg-nrix">0.1209</td>
    <td class="tg-nrix">Precision: </td>
    <td class="tg-nrix">0.7870</td>
  </tr>
  <tr>
    <td class="tg-nrix">F1 Score: </td>
    <td class="tg-nrix">0.1868</td>
    <td class="tg-nrix">F1 Score: </td>
    <td class="tg-nrix">0.8154</td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="3">Llama2 c.</td>
    <td class="tg-nrix" rowspan="3">0.3285</td>
    <td class="tg-nrix">Recall: </td>
    <td class="tg-nrix">0.5029</td>
    <td class="tg-nrix" rowspan="3">0.2992</td>
    <td class="tg-nrix">Recall: </td>
    <td class="tg-nrix">0.8643</td>
    <td class="tg-nrix" rowspan="3">0.8310</td>
  </tr>
  <tr>
    <td class="tg-nrix">Precision: </td>
    <td class="tg-nrix">0.1619</td>
    <td class="tg-nrix">Precision: </td>
    <td class="tg-nrix">0.7988</td>
  </tr>
  <tr>
    <td class="tg-nrix">F1 Score: </td>
    <td class="tg-nrix">0.2328</td>
    <td class="tg-nrix">F1 Score: </td>
    <td class="tg-nrix">0.8300</td>
  </tr>
</tbody>
</table>

Using my dataset seems to have negatively affected the model. Especially for code. Degradation in the coding skill was to be expected as the datasets are very different. While I expected that the fine-tuning would have some effect on the ability of the model to create code I didn't realize how much. Out of the 20 prompts used for generation little code was produced. Even the code that was produced ended up being gibberish and not that good. This cause the CodeBleu score for 2b to be horrible though the original model and the 2c model could still produce code. Surprisingly though it did well on the Rouge and Bert scores.

**3b.) Create 10 instructions completely out-of-sample from your dataset, which produces good results on the original pre-trained model. Use these instructions to generate inference from the original pre-trained model, and the model you saved in 2b and 2c. Write a comparison analysis on the outcome from various stages of trained models.**
To get the out of sample instructions, I asked Gemini and ChatGPT to provide some sample prompts for an LLM. It gave a well-rounded group of instructions including a translation to Spanish and prompts to create poems, Haikus, songs, movie scripts. and jokes.

In my base mode, the prompt to **"Translate this sentence into Spanish: 'The quick brown fox jumps over the lazy dog."** generated a particularly good inference. The model gave:

Translate this sentence into Spanish: 'The quick brown fox jumps over the lazy dog.' Translating the sentence into Spanish... 'El rapaz veloz y listo salta sobre el perro dormilón'
Translated sentence: 'The quick brown fox jumps over the lazy dog.'

Initiating the process, let's achieve perfection! The translation of the sentence in Spanish is:

*'El rapaz veloz y listo salta sobre el perro dormilón.'*

*The translation is:*

*'The quick brown fox jumps over the lazy dog.'*

*The words 'rapaz' and 'perro' are masculine, while 'veloz' and 'dormilón' are adjectives that describe the nouns 'rapaz' and 'perro' respectively. The adjective 'veloz' is an antonym of 'dormilón' which means 'lazy'. The adjective 'veloz' means 'quick' and the adjective 'rapaz' means 'brown'. The noun 'rapaz' means 'fox' and the noun 'perro' means 'dog'.*

This is actually wrong. See 'rapaz' actually means 'raptor' So in actuality the code translated it to:

*'The quick and clever raptor jumps on the sleeping dog.'* Which is a shame for the dog.

When using the IMDB dataset and fine tuning with it something interesting happens. the result this time was:

./Llama2b output:
Translate this sentence into Spanish: 'The quick brown fox jumps over the lazy dog.'

Answer: 'La fiera rápida negra saltó sobre el perro manso.'

Answer: 'La fiera rápida negra saltó sobre el perro manso.'

Comment: Welcome to Spanish.SE! Your answer would be much improved by providing a translation and some explanation of how the sentence is translated.

Now this is still wrong. The answer it gave actually translates too *'The fast black beast jumped on the tame dog.'* So while it gave the wrong color it at the very least is no longer confusing *rapaz* for fox.

After finetuning with the combined dataset, the 3c model ended up translating it to this:

*Instruction 9
*Translate this sentence into Spanish: 'The quick brown fox jumps over the lazy dog.'
***
*./Llama2c output:
*Translate this sentence into Spanish: 'The quick brown fox jumps over the lazy dog.' Translating the sentence... 'El rapaz listo y veloz salta sobre el perro perezoso.' Translated the sentence. Translating... ```python
*import pyttsx3
*engine = pyttsx3.init()
*engine.say('El rapaz listo y veloz salta sobre el perro perezoso.')
*engine.runAndWait()
*```*

This translates to *The clever and fast raptor jumps over the lazy dog.*

It is still wrong but hey at least the dog survived!

In case you’re wondering the phrase, it should have translated to was *El veloz zorro marrón salta por encima del perro perezoso.* or *El rápida zorro marrón salta por encima del perro perezoso.*

If I had to assume it's probable that my dataset was too specific causing it to be confused by such a specific instruction that required more information. The original dataset was:

  **1)** A lot bigger(36K rows vs a little more than 400 rows)
  
  **2)** Much more varied.

It is interesting to see how much fine-tuning can affect LLMs. Fine tuning allowed with just a handful of data rows, allowed me to influence the Llama's translation capabilities.
