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
    pdfFileObj1 = open('water1.pdf','rb')

    pdfReader1 = PyPDF4.PdfFileReader(pdfFileObj1)

    pages1 = pdfReader1.numPages

    for i in range(pages1):
            pageObj1 = pdfReader1.getPage(i)
            text1 = pageObj1.extractText().split("  ")

    pdfFileObj1.close()


    text1=list()
    for line1 in text1:
        line1 = line1.replace('\n','')
        text1.append(line1)


    list1_joined1 = ",".join(text1)
    
    pdfFileObj2 = open('water2.pdf','rb')

    pdfReader2 = PyPDF4.PdfFileReader(pdfFileObj2)

    pages2 = pdfReader2.numPages

    for i in range(pages2):
            pageObj2 = pdfReader2.getPage(i)
            text2 = pageObj2.extractText().split("  ")

    pdfFileObj2.close()


    text2=list()
    for line2 in text2:
        line2 = line2.replace('\n','')
        text2.append(line2)


    list1_joined2 = ",".join(text2)


    ##########################

    corpus=[]
    corpus.append(list1_joined1)
    corpus.append(list1_joined2)


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
