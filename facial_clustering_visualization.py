import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from openpyxl import load_workbook

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


def transform_indices(cell):
    """
    Get indices of the pictures from the cell value
    :param cell: cell that include string in format "bs(2)bs(3)"
    :return: names of the pictures mentioned in the cell (in the example - bs(2), bs(3) )
    """
    splitted_cell = cell.split(")")
    first_picture = splitted_cell[1] + ")"
    second_picture = splitted_cell[2] + ")"

    return first_picture, second_picture


def resize_images_func(size, path):
    """
    Resize all the pictures in path to sizeXsize
    :param size: the pictures' length and width will be adjusted to sizeXsize
    :param path: to the pictures' folder
    :return: none
    """
    from PIL import Image
    import os
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(f"{path}/{item}"):
            im = Image.open(f"{path}/{item}")
            f, e = os.path.splitext(f"{path}/{item}")
            imResize = im.resize((size, size), Image.ANTIALIAS)
            imResize.save(f"{f}.jpg", 'JPEG', quality=90)
    print("DONE")


def create_dist_mat(filename, sheets):
    """
    Reads the excel and parses it to a collection of distance matrices for each person
    :param filename: excel file contains the information in a format which each sheet is a group.
                     In each tab:
                        Line has data about two compared pictures.
                        First cell contains the picture names (example: NAME(1)NAME(2))
                        NOTICE: should contain all the pairs!
                        TODO: should be changed to handle different names. maybe create a set (pic_name,index) ourselves
                              currently - demands different names will have different numbers
    :param sheets: a set of (sheet names, sheet group size)
    :return: two maps:
    person's name to distance matrix
    person's name to pics names to index (array)
    """
    wb = load_workbook(filename=filename)  # every person has a tab
    total_matrices = {}
    total_names_to_indices = {}
    for sheet in sheets:
        names_to_index = []
        # init an empty distance matrix of size (sheet group size)X(sheet group size)
        dist_matrix = [[0] * sheet[1] for i in range(sheet[1])]
        first = True
        for row in wb[sheet[0]].rows:  # for each row (skip the first row)
            if first:
                first = False
                continue
            pic1, pic2 = transform_indices(row[0].value)
            # init a name to index mapping (an array of size sheet group size) for each sheet
            if pic1 not in names_to_index:
                names_to_index.append(pic1)
            if pic2 not in names_to_index:
                names_to_index.append(pic2)
            ind1 = names_to_index.index(pic1)
            ind2 = names_to_index.index(pic2)
            dist_matrix[ind1][ind2] = row[1].value  # set the distance between pic1 and pic2 in the dist matrix
            dist_matrix[ind2][ind1] = row[1].value  # same
        # print(dist_matrix)
        total_matrices[sheet[0]] = dist_matrix  # set the mapping from sheet name to dist matrix
        total_names_to_indices[sheet[0]] = names_to_index  # set the mapping from sheet name to it's pics to index array
    return total_matrices, total_names_to_indices


# a plotting function to plot the result of TSNE dimensional reduction
# inputs:
#       x =  a two dimensional array, X[:,0] is x_axis, X[:,1] is y_axis
#       picture_names = the names of the pictures that correspond to each x,y point in the plot
#       name_person = a specific name of a person (Blanca, Carlos, etc)
def tsne_scatter(x, pictures_folder_path, index_to_picture_name, title):
    # imports relevant for plotting an image in the graph
    from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
    import matplotlib.image as mpimg

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40)
    ax.axis('on')
    ax.set_facecolor('xkcd:white')
    ax.grid(color='xkcd:light grey')
    xmin, xmax, ymin, ymax = -500, 500, -500, 500  # TODO: check if these limits are enough

    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax.set_title(f'TSNE for {title}', fontsize=10)
    for index in index_to_picture_name.keys():
        pic_name = index_to_picture_name[index]
        image_path = f'{pictures_folder_path}/{pic_name}'

        arr_img = mpimg.imread(image_path)  # open the image

        imagebox = OffsetImage(arr_img, zoom=0.25)  # create an image box with a certain zoom

        ab = AnnotationBbox(imagebox, xy=(x[:, 0][index], x[:, 1][index]), pad=0.07)  # create an annotation box, which is at XY on graph

        ax.add_artist(ab)

    plt.savefig(f"{title}.png")  # save the figure to the current directory

    plt.show()  # opens a window with the plot
    return (f, ax, sc)


# distance_matrices, name_to_picture_names = create_dist_mat('Similarity_Ratings_human_fixed.xlsx', names)


# a main function to iterate over all persons and plot the tsne visualization
def main():
    return 0


women_names = ['Blanca', 'Franka', 'Giovanna', 'Johanna', 'Noomi']
men_names = ['Carlos', 'Francesco', 'Guillame', 'Lambert', 'Stefano']


def plot_grid_faces_per_face(data_df, num_faces, pictures_folder_path, query_face):
    import matplotlib.image as mpimg

    closest_faces = data_df.nsmallest(num_faces, query_face)
    faces = closest_faces.index.values

    fig, ax = plt.subplots(ncols=num_faces, figsize=(5, 5))
    fig.subplots_adjust(hspace=0, wspace=0)

    for j in range(num_faces):
        img = mpimg.imread(f'{pictures_folder_path}/{faces[j]}')
        ax[j].xaxis.set_major_locator(plt.NullLocator())
        ax[j].yaxis.set_major_locator(plt.NullLocator())
        ax[j].imshow(img, cmap="bone")
    plt.title(f"The {num_faces} most similar faces are: \n", fontsize=12)
    plt.show()


def visualize(excel_filename, sheet_name, pictures_folder_path, visualization_method, **kwargs):
    """
    Plots a visualization based on the distances given in the excel file.
    The visualization is chosen by the "visualization_method".
    :param excel_filename: the filename of the excel file containing distances between all faces relevant for analysis.
                           Each sheet - should contain the data for each subgroup,
                           Each line - should be in the following format: Pic1 Pic2 | distance
    :param sheet_name: the sheet we want to plot
    :param pictures_folder_path: the path to the folder containing a folder for each subgroup.
                                 Each subfolder contains the pictures of the faces - to use in the plot.
                                 The names of the pictures should correspond to the excel data (e.g Pic1 and Pic2)
    :param visualization_method: one of the following options:
                                 1.'tsne' - shows a tsne plot based on distances
                                 2. 'grid-view' - by default, shows for each face a grid of 5 most similar faces.
                                 3. 'rdm' - shows a RDM  matrix
                                 4. 'heatmap' - shows a heatmap (similarity measure for each pair of faces in each subgroup)
    :param kwargs: optional parameters to specify for certain visualization methods.
           1. num_faces: used in grid-view method- how many faces to display for query_face
           2. query_face: used in grid-view method- for which face to show most similar faces?
    :return: None
    """
    import pandas as pd

    # create a dataframe from the excel_filename and sheet_name
    df = pd.read_excel(excel_filename, sheet_name=sheet_name)  # dependency- xlrd
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    labels_dic = {}
    for i in range(df.shape[0]):
        labels_dic[i] = df.columns[i]
    df = df.rename(labels_dic, axis='index')
    # print(df)
    # print(labels_dic)

    if visualization_method == 'rdm':
        f, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(df, ax=ax, xticklabels=True, yticklabels=True)
        ax.set_title(f'RDM for {sheet_name}', fontsize=10)  # Set the title
        ax.tick_params(axis='x', labelsize=5)  # change size of x-axis
        ax.tick_params(axis='y', labelsize=5)  # change size of y-axis
        # plt.xticks(rotation=30)
        # plt.yticks(rotation=10)
        f.savefig(f'RDM for {sheet_name}.png', dpi=100, bbox_inches='tight')  # TODO: save figure
        # plt.show()  # show plt

    if visualization_method == 'grid-view':
        num_faces_to_plot = kwargs.get('num_faces', 5)
        query_face = kwargs.get('query_face', 'BS_0.png')
        plot_grid_faces_per_face(data_df=df,
                                 pictures_folder_path=pictures_folder_path,
                                 num_faces=num_faces_to_plot,
                                 query_face=query_face)

    if visualization_method == 'tsne':
        tsne_object = TSNE(method="exact", metric="precomputed", random_state=RS, perplexity=2)
        fashion_tsne = tsne_object.fit_transform(df)
        tsne_scatter(fashion_tsne, pictures_folder_path=pictures_folder_path,  index_to_picture_name=labels_dic, title=sheet_name)


visualize('openface_dists - test.xlsx', pictures_folder_path=f'faces_for_tsne_visualization/women',
          sheet_name='women', visualization_method='grid-view')


