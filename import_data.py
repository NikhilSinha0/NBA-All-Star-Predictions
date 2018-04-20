import csv
import random
import copy

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

def append_allStar_data(fileName, data, year):
    f=open(fileName, "r")
    contents = f.read()
    names = contents.split("\n")
    for item in data:
        item['ASG'] = (item['Player'] in names)
        item['Player'] = item['Player'] + ' ' + str(year)

def extract_ASG_vector(data):
    asg_vector = []
    names_vector = []
    for item in data:
        names_vector.append(item['Player'])
        asg_vector.append(1 if (item['ASG']) else 0)
        del item['Player'] #No longer need player names, do not want to learn from name
        del item['ASG'] #Extract vector of all stars and make sure we do not learn from whether the player made the all star game that year
    return asg_vector, names_vector

def over_sample_data(data):
    all_stars = []
    for item in data:
        if(item['ASG']):
            all_stars.append(copy.deepcopy(item))
    imbalance = len(data)//len(all_stars)
    print(imbalance)
    for i in range(imbalance):
        data.extend(copy.deepcopy(all_stars))
    return data

def get_cleaned_data(data_file, asg_file, year):
    data = get_player_data(data_file)
    data = clean_data(data)
    append_allStar_data(asg_file, data, year)
    print("Import done")
    return data

def get_data(data_files):
    features = []
    for item in data_files:
        data_file = item[0]
        asg_file = item[1]
        year = item[2]
        this_features = get_cleaned_data(data_file, asg_file, year)
        features += this_features
    random.shuffle(features) #Randomize order of data
    labels, names = extract_ASG_vector(features) #Extract target labels from randomized data
    return features, labels, names

def get_oversampled_data(data_files):
    features = []
    for item in data_files:
        data_file = item[0]
        asg_file = item[1]
        year = item[2]
        this_features = get_cleaned_data(data_file, asg_file, year)
        features += this_features
    features = over_sample_data(features) #Oversample data
    random.shuffle(features) #Randomize order of data
    labels, names = extract_ASG_vector(features) #Extract target labels from randomized data
    return features, labels, names