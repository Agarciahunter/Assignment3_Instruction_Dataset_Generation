from peft import PeftModel, PeftConfig
from datasets import load_dataset
import random
from codebleu import calc_codebleu
from rouge import Rouge
from bert_score import score
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    StoppingCriteriaList,
    MaxLengthCriteria,
    )

def getOutput(tokenizer,model,testPrompt,hparam,size=0):
    input = tokenizer(testPrompt, return_tensors="pt").input_ids
    input = input.to('cuda')
    if hparam == "vanilla":
        # Generate output using vanilla decoding
        outputs = model.generate(input, max_length = 450)
    elif hparam == "topK":
        # Generate output using top-K sampling
        outputs = model.generate(input,
                                 max_length = 450,
                                 do_sample=True,
                                 top_k=size)
    elif hparam == "beam":
        # Generate output using beam search
        outputs = model.generate(input,
                                 max_length = 450,
                                 num_beams=size,
                                 early_stopping=True)
    elif hparam == "temp":
         # Generate output using temperature sampling
         outputs = model.generate(input,
                                 max_length = 450,
                                 do_sample=True,
                                 top_k=0,
                                 temperature = size)
    # Generate output at different model layers
    elif hparam == "layer":
        logits_processor = LogitsProcessorList(
        [
        MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
        ]
        )
        # instantiate logits processors
        logits_warper = LogitsProcessorList(
        [
            TopKLogitsWarper(50),
            TemperatureLogitsWarper(0.7),
        ]
        )

        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=50)])

        torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        outputs = model._dola_decoding(
            input,
            dola_layers=[size],
            max_length = 450,
            repetition_penalty=1.2,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            # output_scores=True,
            # return_dict_in_generate=True,
        )


    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


modelList = [
    "./LlamaBase",
    "./Llama2b",
    "./Llama2c",
]

outputType = [
    "vanilla",
    #"topK",
    #"beam",
    #"temp",
    #"layer"
]

topKsize = [
    2,
    4,
    6,
    8
]

beamsize = [
    2,
    3,
    4,
    5
]

tempSize = [
    .1,
    .25,
    .5,
    .75
]

layernum = [
    8,
    16,
    24,
]

datapath = "flytech/python-codes-25k"

# Load dataset

randrows = []
referencelist = []
# Generated from Google Gemini
testPrompts = ["Write a short story in the style of a movie trailer script, using only emojis to convey the plot.",
               "Write a short poem in the style of Haiku (three lines, five syllables, seven syllables) about a robot who dreams of becoming a chef.",
               "Come up with a new, creative use for a standard household object, like a colander or a whisk.",
               "Compose a limerick, a humorous poem with a specific rhyme scheme (AABBA), based on a randomly generated topic. The limerick should maintain the five-line structure and incorporate humor.",
               "Can you explain the concept of quantum computing in simple terms?",
               "Write a review for a local bakery, focusing on the ambiance and customer service rather than the taste of the food.",
               "Write a song from the perspective of a historical artifact, detailing its journey through time.",
               "Write a fictional news headline from the year 2524 describing a major breakthrough in interstellar travel.",
               "Write a joke about the struggles of being a AI in today's society.",
               "Translate this sentence into Spanish: 'The quick brown fox jumps over the lazy dog.'"]
# Generated from ChatGPT
referencelist = ["Earth: A once peaceful planet, now on the brink of destruction. A team of unlikely heroes must band together to save humanity. Explosions, battles, and heart-pounding action await as they journey across the galaxy. Alien adversaries, ancient artifacts, and a race against time. Will they succeed, or will Earth face its ultimate demise? Get ready for the adventure of a lifetime in 'Galactic Saviors' - coming soon to a theater near you.",
                 "Metallic hands stir,\nDreams of flavors in circuits,\nChef in gears, it dreams.",                 
                 "Transform a colander into a hanging planter! With its perforated design, it allows for proper drainage while adding a unique touch to your indoor garden. Just suspend it from the ceiling, fill it with soil, and plant your favorite herbs or small flowers. It's both functional and decorative, bringing a touch of greenery into your home in a surprising way.",                 
                 "There once was a chicken named Lou,\nWho dreamed of a life with a view.\nHe built a coop high,\nBut then wondered why,\nHis wings couldn't get him there too!",
                 "Sure! Imagine you have a regular computer, like the one you're using right now. It uses bits to process information. These bits can either be a 0 or a 1, like a light switch being either off or on. Now, picture a quantum computer. Instead of bits, it uses quantum bits, or qubits. Here's where things get interesting. Unlike regular bits, qubits can be both 0 and 1 at the same time. It's like having a light switch that can be in multiple positions simultaneously! This ability of qubits to exist in multiple states simultaneously is due to a property called superposition. Additionally, qubits can be entangled, meaning the state of one qubit can instantly affect the state of another, no matter the distance between them. These properties of superposition and entanglement allow quantum computers to perform certain calculations much faster than classical computers. They're especially good at solving problems involving lots of possibilities or searching through vast amounts of data. So, in simple terms, quantum computing is a new way of computing that harnesses the strange and powerful principles of quantum mechanics to perform tasks that would be practically impossible or extremely time-consuming for classical computers.",
                 "Stepping into [Bakery Name] is a treat for the senses even before you reach the pastry counter. The aroma of warm bread and buttery pastries hits you first, a delightful invitation that promises fresh-baked goodness. The bakery itself is charming, with [describe the decor - exposed brick, vintage furniture, modern and airy, etc.].  [Mention any visual details - chalkboard menu, display case overflowing with goodies, etc.].  There's a real sense of warmth and care put into the space, making you feel instantly welcome. \nThe staff at [Bakery Name] adds to the delightful experience. From the moment you walk in, you're greeted with a friendly smile and a genuine 'hello.' The staff is incredibly patient, especially with those facing the overwhelming decision of which pastry to choose first. They're knowledgeable about the menu and happy to answer any questions or offer recommendations. \nWhether you're grabbing a quick coffee and a croissant to-go or settling in for a leisurely breakfast, [Bakery Name] provides a delightful ambiance and top-notch customer service. It's the perfect spot to relax, savor a delicious treat, and feel right at home.",
                 "(Verse 1) Inscribed in stone, I was born in ancient lands, A witness to empires rise and fall, grains of sand. Carved by skilled hands, with stories to impart, I'm a relic of the past, with a beating heart. \n(Chorus) I've seen the world change, felt the winds of time, A silent sentinel, in this grand design. Through wars and peace, I've stood tall and strong, A testament to the echoes of history's song. \n(Verse 2) From the pyramids of Egypt to Rome's grand halls, I've seen mighty kingdoms rise, only to crumble and fall. Through the ages I've traveled, across lands and seas, A symbol of resilience, in the face of destinies. \n(Chorus) I've seen the world change, felt the winds of time, A silent sentinel, in this grand design. Through wars and peace, I've stood tall and strong, A testament to the echoes of history's song. \n(Bridge) I've felt the touch of kings, the hands of common men, Each mark and inscription, a tale to transcend. Though I may weather and age, my spirit remains, For I am more than just stone, I am history's refrain. \n(Chorus) I've seen the world change, felt the winds of time, A silent sentinel, in this grand design. Through wars and peace, I've stood tall and strong, A testament to the echoes of history's song. \n(Outro) So as the ages roll on, and civilizations rise and fall, I'll continue to stand, a guardian of it all. For I am not just an artifact, but a keeper of the past, A living legacy, destined to forever last.",
                 "Breakthrough in Quantum Entanglement Propulsion Unlocks Interstellar Travel: Humanity Ventures Beyond the Stars",
                 "Why did the AI cross the road? \nBecause its algorithms couldn't decide whether to take the shortest route or the most scenic one, so it ended up stuck in traffic, endlessly recalculating!",
                 "Sure! The translation of the sentence 'The quick brown fox jumps over the lazy dog' into Spanish is: 'El rápido zorro marrón salta sobre el perro perezoso.'",
            ]
numInputs = 10
# print(len(referencelist))
# print(len(testPrompts))

for modelpath in modelList:
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(modelpath)
    model = PeftModel.from_pretrained(model, modelpath)
    tokenizer = AutoTokenizer.from_pretrained(modelpath)

    for hparam in outputType:
        sizes = []
        if hparam == "vanilla":
            sizes = [1]
        elif hparam == "topK":
            sizes = topKsize.copy()
        elif hparam == "beam":
            sizes = beamsize.copy()
        elif hparam == "temp":
            sizes = tempSize.copy()
        elif hparam == "layer":
            sizes = layernum.copy()

        for size in sizes:
            predictionlist = []
            for i in range(numInputs):
                print("Getting output for: " + str(modelpath)+ "... output type: "+ str(hparam)+ " size = "+ str(size) + "...Instruction:" + str(i+1))
                # testPrompt = dataset[i]["instruction"]
                
                text = getOutput(tokenizer,model,testPrompts[i],hparam,size)
                
                # referencelist.append(dataset[i]["output"])
                predictionlist.append(text)
            
            print("Results for " + str(modelpath)+ "... output type: "+ str(hparam)+ " size = "+ str(size))
            print('-' * 80)
            ##codebleu##
            codebleuResult = calc_codebleu(referencelist, predictionlist, lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
            print("CodeBleu Scrore: " + str(codebleuResult["codebleu"]))
            ##rouge##
            rouge = Rouge()
            scores = rouge.get_scores(predictionlist, referencelist, avg=True)
            print("Rouge-L score: " + str(scores["rouge-l"]))
            ##BERTscore##
            P, R, F1 = score(predictionlist, referencelist, lang="en", verbose=True)
            recall = R.mean().item()
            precision = P.mean().item()
            f1_score = F1.mean().item()
            print("BERTScore:")
            print(P, R, F1)
            bert_score_str = f"r: {recall}, p: {precision}, f: {f1_score}"
            print("BertScore (P, R, F1)")
            print(bert_score_str)


            print('-' * 80)
            print("")

            print("For Human Evaluation on : " + str(modelpath)+ "... output type: "+ str(hparam)+ " size = "+ str(size))
            with open("evaluation_metrics3B.txt", "a") as file:
              file.write(f"Model: {modelpath}, Output Type: {hparam}, Size: {size}\n")
              file.write(f"CodeBleu Score: {codebleuResult['codebleu']}\n")
              file.write(f"Rouge-L score: {scores['rouge-l']}\n")
              file.write(f"BertScore: {bert_score_str}\n")
              file.write("-" * 80 + "\n")
              
            print("For Human Evaluation on : " + str(modelpath)+ "... output type: "+ str(hparam)+ " size = "+ str(size))
            
            #only need 20 for human evaluation
            if numInputs > 20:
                numHumanEval = 20
            else:
                numHumanEval = numInputs

            for i in range(numHumanEval):
                print("Instruction " + str(i))
                
                print(testPrompts[i])
                print("***")
                print(str(modelpath) + " output:")
                print(predictionlist[i])
                print('-' * 80)