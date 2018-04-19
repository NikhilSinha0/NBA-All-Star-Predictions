import csv
import random

def get_player_data(fileName):
    data = []
    with open(fileName, 'rt') as f:
        reader = csv.reader(f)
        vals = []
        isFirst = True
        for row in reader:
            if(isFirst):
                for i in range(len(row)):
                    vals.append(row[i])
                isFirst = False
            else:
                item = {}
                for i in range(len(row)):
                    val = row[i]
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            val = str(val)
                    if(vals[i]=='Rk'): #Skip field with no real information
                        continue
                    item[vals[i]] = val
                data.append(item)
    return data
            
def clean_data(data):
    indices_to_delete = []
    for i in range(1, len(data)):
        if(data[i]['Player'] == data[i-1]['Player']):
            indices_to_delete.append(i) #Delete repeat entries and keep only totals
        
    data = [data[e] for e in range(len(data)) if e not in indices_to_delete]
    return data

def append_allStar_data(fileName, data):
    f=open(fileName, "r")
    contents = f.read()
    names = contents.split("\n")
    for item in data:
        item['ASG'] = (item['Player'] in names)
        del item['Player'] #No longer need player names, do not want to learn from name

def extract_ASG_vector(data):
    asg_vector = []
    for item in data:
        asg_vector.append(1 if (item['ASG']) else 0)
        del item['ASG'] #Extract vector of all stars and make sure we do not learn
    return asg_vector   #from whether the player made the all star game that year

def get_cleaned_data(data_file, asg_file):
    data = get_player_data(data_file)
    data = clean_data(data)
    append_allStar_data(asg_file, data)
    print("Import done")
    return data

def get_data(data_files):
    features = []
    for item in data_files:
        data_file = item[0]
        asg_file = item[1]
        this_features = get_cleaned_data(data_file, asg_file)
        features += this_features
    random.shuffle(features) #Randomize order of data
    labels = extract_ASG_vector(features) #Extract target labels from randomized data
    return features, labels