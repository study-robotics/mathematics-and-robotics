import json

data = {}

total_data_num = 0

data_num = 30000
val_data_num = int(data_num*0.2)
data["val"] = []
for i in range(total_data_num, total_data_num+val_data_num):
    data["val"].append(i)
total_data_num = total_data_num+val_data_num

train_data_num = int(data_num*0.8)
data["train"] = []
for i in range(total_data_num, total_data_num+train_data_num):
    data["train"].append(i)
total_data_num = total_data_num+train_data_num

test_data_num = 50
test_data = [
    15, 30, 45, 75, 90, 105, 120, 135, 165, 180, 195, 
    225, 255, 270, 285, 315, 360, 375, 405, 495, 510, 
    540, 555, 585, 615, 645, 735, 750, 765, 780, 810, 
    825, 870, 885, 900, 915, 930, 960, 1005, 1020, 1035,
    1080, 1095, 1110, 1155, 1170, 1215, 1245, 1260, 1290]
data["test"] = []
for i in range(test_data_num):
    data["test"].append(test_data[i])
total_data_num = total_data_num+test_data_num
list_data = []
list_data.append(data)
with open('./vox_cspace.json', 'w') as f:
    json.dump(list_data, f, indent=4, ensure_ascii=False)
