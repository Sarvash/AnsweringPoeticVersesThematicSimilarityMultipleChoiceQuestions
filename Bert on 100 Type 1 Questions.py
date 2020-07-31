import xlrd
from bert_serving.client import BertClient
import numpy as np

bc = BertClient(check_length=False)
location = (r"/home/sarvash/Dropbox/Sharif University/Thesis/Baseline/Type 1 Questions.xlsx")
wb = xlrd.open_workbook(location, encoding_override='utf-8')
sheet = wb.sheet_by_index(0)
correct_answers = 0
top_two = 0
for row in range(1, 101):
    options = sheet.row_values(row, 1, 5)
    options_vec = bc.encode(options)
    stem = sheet.cell_value(row, 0)
    stem_vec = bc.encode([stem])[0]
    score = np.sum(stem_vec * options_vec, axis=1) / np.linalg.norm(options_vec, axis=1)
    print(score)
    top_index = np.argsort(score)[::-1]
    answer_prediction = top_index[0] + 1
    second_prediction = top_index[1] + 1
    answer = int(sheet.cell_value(row, 5))
    print("The predicted answer is " + str(answer_prediction) + ". The correct answer is " + str(answer) + ".")
    if answer_prediction == answer:
        correct_answers += 1
        top_two += 1
    elif second_prediction == answer:
        top_two += 1
    # Printing verses in order of score with their associative score
    for index in top_index:
        print('> %s\t%s' % (score[index], options[index]))
    print("\n")
    print("-------------------------------------")
    print("\n")

print("Model Accuracy: " + str(correct_answers))
print("Answer in Top 2 Predictions: " + str(top_two))