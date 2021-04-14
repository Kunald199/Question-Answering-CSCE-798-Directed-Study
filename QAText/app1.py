from flask import Flask,render_template,request
import torch
import PyPDF4
from rank_bm25 import BM25L
from transformers import BertForQuestionAnswering  #pretrained model
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
from transformers import BertTokenizer  #loading tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
import pickle



app= Flask(__name__)

@app.route('/',methods = ['POST', 'GET'])
def hello():
    return render_template("after.html")

@app.route('/predict',methods = ['POST', 'GET'])
def home():
    pdfFileObj = open('Water.pdf','rb')

    pdfReader = PyPDF4.PdfFileReader(pdfFileObj)

    pages = pdfReader.numPages

    for i in range(pages):
            pageObj = pdfReader.getPage(i)
            text = pageObj.extractText().split("  ")

    pdfFileObj.close()


    text1=list()
    for line in text:
        line = line.replace('\n','')
        text1.append(line)


    list1_joined1 = ",".join(text1)



    ##########################


    corpus=[]
    corpus.append(list1_joined1)







    # Trying for BM25
    """corpus=["Sandra Day O’Connor, née Sandra Day, (born March 26, 1930, El Paso, Texas, U.S.), associate justice of the Supreme Court of the United States from 1981 to 2006. She was the first woman to serve on the Supreme Court. A moderate conservative, she was known for her dispassionate and meticulously researched opinions. Sandra Day grew up on a large family ranch near Duncan, Arizona. She received undergraduate (1950) and law (1952) degrees from Stanford University, where she met the future chief justice of the United States William Rehnquist.",
        ''' In South Carolina,the action level for lead in drinking water is 0.015 mg/L and action level for copper in drinking water is 1.3 mg/L.
        In North Carolina,the action level for lead in drinking water is 0.015 mg/L and action level for copper in drinking water is 1.3 mg/L.
     In New York,the action level for lead in drinking water is 0.015 mg/L and action level for copper in drinking water is 1.3 mg/L.
     The Environmental Protection Agency (EPA) kept safe level for lead in drinking water as 0.015 mg/L safe level for copper in drinking water as 1.3 mg/L.
     The World Health Organization action level for lead in drinking water is 0.01 mg/L and action level for copper in drinking wateris 2 mg/L.
     In Michigan,the action level for lead in drinking water is 0.015 mg/L and action level for copper in drinking water is 1.2 mg/L.
     California has set action level of lead to 0.015 mg/L and for copper the action level is 0.0013 mg/L.''',
     '''The Fourth Amendment of the U.S. Constitution provides that the right of the people to be secure in their persons, houses, papers, and effects, against unreasonable searches and seizures, shall not be violated, and no Warrants shall issue, but upon probable cause, supported by Oath or affirmation, and particularly describing the place to be searched, and the persons or things to be seized.'The ultimate goal of this provision is to protect people’s right to privacy and freedom from unreasonable intrusions by the government. However, the Fourth Amendment does not guarantee protection from all searches and seizures, but only those done by the government and deemed unreasonable under the law.''']"""



    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25L(tokenized_corpus)
    question=request.form.get('a')

    #question='What is safe limit for copper according to South Carolina?'
    tokenized_query = question.split(" ")

    doc=bm25.get_top_n(tokenized_query, corpus, n=1)
    answer_text=doc[0]

    #answer_text1=bm25.get_top_n(tokenized_query, corpus, n=1)
    #answer_text=answer_text1[0]

    #print(answer_text)
    ##############################

    input_ids = tokenizer.encode(question, answer_text)  #applying tokenizer to get total tokens



    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # See where the SEP token is
    sep_index = input_ids.index(tokenizer.sep_token_id)


    SEG_A_num = sep_index + 1


    SEG_B_num = len(input_ids) - SEG_A_num

    #list of 0 and 1
    segment_ids = [0]*SEG_A_num + [1]*SEG_B_num

    #Ensure for each input id we have segment id and of same length
    assert len(segment_ids) == len(input_ids)





    start, end = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                     token_type_ids=torch.tensor([segment_ids]),return_dict=False) # The segment IDs to differentiate question from answer_text



    beginAnswer = torch.argmax(start)
    endAnswer = torch.argmax(end)
    answer = tokens[beginAnswer]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(beginAnswer + 1, endAnswer + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]




    return render_template("answer.html",data=answer)



if __name__ == "__main__":
    app.run(debug=True)
