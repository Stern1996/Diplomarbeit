import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize the database enviroment
# read in the database
with open("data/results.json") as json_file:
    data = json.load(json_file)
with open("data/criterion.json") as json_file:
    criterion = json.load(json_file)
    print("The criterions-plans include:",criterion.keys())

while True:
    criterion_name = input("Please input the to be analysed criterion-plan:\n")
    if criterion_name in list(data.keys()):
        data = data[criterion_name]
        # read in the criterion
        with open("data/criterion.json") as json_file:
            criterion = json.load(json_file)
        print("The criterion is:\n", criterion[criterion_name])
        break
    else:
        print("There is no the result of criterion-plan:\n", criterion_name)
# read in the datasets-properties
with open("data/index.json") as json_file:
  data_index = json.load(json_file)


#Define the necessary funtions
# return the datasets properties of the given feature
def get_filename(data_index,tags):
    file_names = set()
    error_type = set()
    frequency_type = set()
    size_type = set()
    for file in data_index:
        if file_has_tags(data_index[file], tags):
            file_names.add(file)
            error_type.add(data_index[file]["error"])
            frequency_type.add(data_index[file]["frequency"])
            size_type.add(data_index[file]["size"])
    return file_names, error_type, frequency_type, size_type

# determin if a file as all the tags
def file_has_tags(json, tags):
    flag = True
    for tag in tags:
        if json["frequency"] != tag and json["size"] != tag and json["error"] != tag:
            flag = False
    return flag

# find the file of given tags
def find_file(tags):
  for file in data_index:
    if(set(tags) == set(data_index[file].values())):
      return file
  return None

#get the necessary information of diagram
def get_diagramm_data(constant_feature):
  file_names, error_type, frequency_type, size_type = get_filename(data_index,[constant_feature])
  features = {"error":error_type,"frequency":frequency_type,"size":size_type}
  for i in features:
    if len(features[i]) == 1:
      del features[i]
      break
  x_label = list(features.keys())[0]
  y_label = list(features.keys())[1]
  x_value = list(features[x_label])
  y_value = list(features[y_label])
  z_value = np.zeros(shape=(len(x_value),len(y_value)))
  for i in range(len(x_value)):
    for j in range(len(y_value)):
      tags = [constant_feature,x_value[i],y_value[j]]
      if find_file(tags) != None:
        z_value[i][j] = data[find_file(tags)]
  diagramm_data = [{x_label:x_value},{y_label:y_value},{"accuracy":z_value}]
  return diagramm_data


# Drawing
def draw_3dbar(data, constant_feature):
    # parse the diagram-information
    for index in range(len(data)):
        if list(data[index].keys())[0] == "error":
            error_info = []
            for i in data[index]["error"]:
                if i == "no":
                    error_info.append(0)
                elif i == "ir":
                    error_info.append(1)
                elif i == "ba":
                    error_info.append(2)
                elif i == "or":
                    error_info.append(3)
            data[index] = {"error": error_info}

    x_label = list(data[0].keys())[0]
    y_label = list(data[1].keys())[0]
    z_label = list(data[2].keys())[0]
    x_value = np.asarray(data[0][x_label])
    y_value = np.asarray(data[1][y_label])
    z_value = np.asarray(data[2][z_label])

    # Drawing
    xx, yy = np.meshgrid(x_value, y_value)  # meshgrid-coordinates
    X, Y = xx.ravel(), yy.ravel()  # flatten matrix
    bottom = np.zeros_like(X)  # set the bottom-value of bar-diagram
    Z = z_value.ravel()  # flatten matrix

    width = height = 1  # set the width and height of each bar

    # Diagram setting
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.bar3d(X, Y, bottom, width, height, Z, shade=True)
    # axis-label
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.savefig("diagramm/"+criterion_name+"_3D_"+str(constant_feature) + "_results.jpg")
    plt.show()

if __name__ == "__main__":
    while True:
        command = input("drawing the 2D or 3D diagram(2D or 3D):\n")
        if command == "3D":
            constant_feature = input("Please input the constant feature-value:\n")
            try:
                constant_feature = float(constant_feature)
            except:
                pass

            data = get_diagramm_data(constant_feature)
            draw_3dbar(data,constant_feature)
            break
        elif command == "2D":
            break
        else:
            print("You give the wrong command:",command)