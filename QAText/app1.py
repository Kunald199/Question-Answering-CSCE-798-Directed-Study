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
