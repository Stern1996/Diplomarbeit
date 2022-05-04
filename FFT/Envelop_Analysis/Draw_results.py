import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize the database enviroment
# read in the database
with open("./data/results.json") as json_file:
    data = json.load(json_file)
    print("The available criterions-plans include:",data.keys())

while True:
    criterion_name = input("Please input the to be analysed criterion-plan:\n")
    if criterion_name in list(data.keys()):
        data = data[criterion_name]
        # read in the criterion
        with open("./data/criterion.json") as json_file:
            criterion = json.load(json_file)
        print("The criterion is:\n", criterion[criterion_name])
        break
    else:
        print("There is no the result of criterion-plan:\n", criterion_name)
# read in the datasets-properties
with open("./data/index.json") as json_file:
  data_index = json.load(json_file)


#Define the necessary funtions
class Draw_3D():
    # return the datasets properties of the given feature
    def get_filename(data_index,tags):
        file_names = []
        error_type = []
        frequency_type = []
        size_type = []
        for file in data_index:
            if Draw_3D.file_has_tags(data_index[file], tags):
                file_names.append(file)
                error_type.append(data_index[file]["error"])
                frequency_type.append(data_index[file]["frequency"])
                size_type.append(data_index[file]["size"])
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
      file_names, error_type, frequency_type, size_type = Draw_3D.get_filename(data_index,[constant_feature])
      features = {"error":error_type,"frequency":frequency_type,"size":size_type}
      #remove the constant-feature
      for i in features:
        if len(list(set(features[i]))) == 1:
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
          if Draw_3D.find_file(tags) != None:
            z_value[i][j] = data[Draw_3D.find_file(tags)]

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
        xx, yy = np.meshgrid(y_value, x_value)  # meshgrid-coordinates
        X, Y = xx.ravel(), yy.ravel()  # flatten matrix
        bottom = np.zeros_like(X)  # set the bottom-value of bar-diagram
        Z = z_value.ravel()  # flatten matrix

        if min(X.max(),Y.max()) < 1:
            width = height = 0.003  # set the width and height of each bar
        elif min(X.max(),Y.max()) < 10:
            width = height = 0.5  # set the width and height of each bar


        # Diagram setting
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.bar3d(X, Y, bottom, width, height, Z, shade=True)
        # axis-label
        ax.legend()
        if X.max() < 1: #X-axis is size
            ax.set_xlim(0.0, 0.028)
        elif X.max() < 10: #X-axis is error
            ax.set_xlim(-1,4)
        else:
            ax.set_xlim(1710,1817) #X-axix is speed

        if Y.max() < 1: #Y-axis is size
            ax.set_ylim(0.0, 0.028)
        elif Y.max() < 10: #Y-axis is error
            ax.set_ylim(-1,4)
        else:
            ax.set_ylim(1710,1817) #Y-axix is speed
        ax.set_zlim(0.0, 1.0)
        ax.set_title("The constant-feature-value is:"+str(constant_feature))
        ax.set_xlabel(y_label)
        ax.set_ylabel(x_label)
        ax.set_zlabel(z_label)
        plt.savefig("diagramm/"+criterion_name+"_3D_"+str(constant_feature) + "_results.jpg")
        plt.show()

class Draw_2D():
    # find the file of given tags
    def find_file(tags):
        for file in data_index:
            if (set(tags) == set(data_index[file].values())):
                return file
        return None

    # determin if a file as all the tags
    def file_has_tags(json, tags):
        flag = True
        set1 = set([json["frequency"],json["size"]])
        set2 = set([json["frequency"], json["error"]])
        set3 = set([json["error"], json["size"]])
        if set(tags) != set1 and set(tags) != set2 and set(tags) != set3:
            flag = False
        return flag

    # Define the necessary funtions
    # return the datasets properties of the given feature
    def get_filename(tags):
        file_names = []
        error_type = []
        frequency_type = []
        size_type = []
        for file in data_index:
            if Draw_2D.file_has_tags(data_index[file], tags):
                file_names.append(file)
                error_type.append(data_index[file]["error"])
                frequency_type.append(data_index[file]["frequency"])
                size_type.append(data_index[file]["size"])
        return file_names, error_type, frequency_type, size_type

    # get the necessary information of diagram
    def get_diagramm_data(constant_feature1, constant_feature2):
        file_names, error_type, frequency_type, size_type = Draw_2D.get_filename([constant_feature1, constant_feature2])
        features = {"error": error_type, "frequency": frequency_type, "size": size_type}
        for i in features:
            if len(list(set(features[i]))) != 1:
                features = {i: features[i]}
                break

        x_label = list(features.keys())[0]
        x_value = list(features[x_label])
        y_value = np.zeros(shape=(len(x_value),))
        for i in range(len(x_value)):
            tags = [constant_feature1, constant_feature2, x_value[i]]
            if Draw_2D.find_file(tags) != None:
                y_value[i] = data[Draw_2D.find_file(tags)]
        diagramm_data = [{x_label: x_value}, {"accuracy": y_value}]
        return diagramm_data

    # Drawing
    def draw_2dbar(data_2d, constant_feature1, constant_feature2):
        # parse the diagram-information
        for index in range(len(data_2d)):
            if list(data_2d[index].keys())[0] == "error":
                error_info = []
                for i in data_2d[index]["error"]:
                    if i == "no":
                        error_info.append(0)
                    elif i == "ir":
                        error_info.append(1)
                    elif i == "ba":
                        error_info.append(2)
                    elif i == "or":
                        error_info.append(3)
                data_2d[index] = {"error": error_info}

        x_label = list(data_2d[0].keys())[0]
        for i in range(len(data_2d[0][x_label])):
            data_2d[0][x_label][i] = str(data_2d[0][x_label][i])
        x_value = np.asarray(data_2d[0][x_label])
        y_label = list(data_2d[1].keys())[0]
        y_value = np.asarray(data_2d[1][y_label])

        # Diagram setting
        plt.bar(x_value, y_value, width=0.5)
        # axis-label
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.ylim(0,1.5)
        plt.title("feature1:" + str(constant_feature1) + ", feature2:" + str(constant_feature2))
        #show value
        for a, b in zip(x_value, y_value):
            plt.text(a, b, str(b), ha='center', va='bottom', fontsize=11)
        plt.savefig(
            "diagramm/" + criterion_name + "_2D_" + str(constant_feature1) + "_" + str(constant_feature2) + ".jpg")
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

            data = Draw_3D.get_diagramm_data(constant_feature)
            Draw_3D.draw_3dbar(data,constant_feature)
            break
        elif command == "2D":
            constant_feature1 = input("Please input the first constant feature-value:\n")
            constant_feature2 = input("Please input the second constant feature-value:\n")
            features = [constant_feature1, constant_feature2]
            for i in range(len(features)):
                try:
                    features[i] = float(features[i])
                except:
                    pass
            data_2d = Draw_2D.get_diagramm_data(features[0], features[1])
            Draw_2D.draw_2dbar(data_2d, features[0], features[1])
            break
        else:
            print("You give the wrong command:",command)