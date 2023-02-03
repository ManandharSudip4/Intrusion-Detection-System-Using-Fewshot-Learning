import csv
import time

# fields = ['Numeral', 'data1', 'data2']
fields = [53,572,1,1,53,107,53,53,53,0,107,107,107,0,279720.2797,3496.503497,572,0,572,572,0,0,0,0,0,0,0,0,0,0,0,0,0,0,40,40,1748.251748,1748.251748,53,107,71,31.17691454,972,0,0,0,0,0,0,0,0,1,106.5,53,107,0,0,0,0,0,0,1,53,1,107,-1,-1,0,40,0,0,0,0,0,0,0,0,'BENIGN']

# numeral = 0
# while True:
#     with open('My.csv', 'a') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fields)
#         data1 = f"example_data{numeral}"
#         data2 = "example_data2"
#         writer.writerow({'Numeral': numeral, 'data1': data1, 'data2': data2})
#         time.sleep(2)
#     numeral+=1


numeral = 0
while True:
    with open('csv_used/predict.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fields)
        time.sleep(2)
    numeral+=1

