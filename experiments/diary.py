#!/usr/bin/env python
"""Diary to create notebooks and store intermediate results and figures

This class can be used to save the intermediate results of any experiment
run in python. It will create a folder for the specific experiment and
save all the information and images in a structured and sorted manner.
"""
__docformat__ = 'restructedtext en'
import os
import sys
import errno
import csv
import datetime

from functools import partial

try:
    import PIL.Image as Image
except ImportError:
    import Image

__author__ = "Miquel Perello Nieto"
__credits__ = ["Miquel Perello Nieto"]

__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Miquel Perello Nieto"
__email__ = "miquel@perellonieto.com"
__status__ = "Development"

class Notebook(object):
    def __init__(self, name, diary, verbose=False):
        self.name = name
        self.filename = "{}.csv".format(name)
        self.diary = diary
        self.entry_number = 0
        self.verbose = verbose

    def add_entry(self, row):
        general_entry_number = self.diary.increase_entry_number()
        self.entry_number += 1
        if type(row) is dict:
            row = sum([[key, str(value).replace('\n', '\\n')] for key, value in row.items()], [])
        with open(os.path.join(self.diary.path, self.filename), 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|',
                    quoting=csv.QUOTE_NONNUMERIC)
            now = datetime.datetime.now()
            row = [general_entry_number, self.entry_number,
                    now.date().__str__(),
                   now.time().__str__()] + row
            writer.writerow(row)
        if self.verbose:
            print(row)

class Diary(object):

    __DESCR_FILENAME='description.txt'

    def __init__(self, name, path='diary', overwrite=False, image_format='png',
                 fig_format='svg', stdout=True, stderr=True):
        self.creation_date = datetime.datetime.now()
        self.name = name
        self.path = os.path.join(path,name)
        self.overwrite = overwrite

        self.image_format = image_format
        self.fig_format = fig_format
        self.entry_number = 0

        self.all_paths = self._create_all_paths(overwrite)
        self._save_description()

        self.stdout = stdout
        self.stderr = stderr

        if self.stdout:
            self.redirect_stdout(self.path)
        if self.stderr:
            self.redirect_stderr(self.path)

        self.notebooks = {}

    def redirect_stdout(self, path, filename='stdout.txt'):
        sys.stdout = open(os.path.join(path, filename), 'w')

    def redirect_stderr(self, path, filename='stderr.txt'):
        sys.stderr = open(os.path.join(path, filename), 'w')

    def add_notebook(self, name, **kwargs):
        self.notebooks[name] = Notebook(name, self, **kwargs)
        return self.notebooks[name]

    def _create_all_paths(self, overwrite):
        original_path = self.path
        created = False
        i = 0
        while not created:
            while overwrite == False and os.path.exists(self.path):
                self.path = "{}_{}".format(original_path,i)
                i +=1

            self.path_images = os.path.join(self.path, 'images')
            self.path_figures = os.path.join(self.path, 'figures')
            all_paths = [self.path, self.path_images, self.path_figures]
            try:
                for actual_path in all_paths:
                    os.makedirs(actual_path)
                created = True
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise
        return all_paths

    def _save_description(self):
        with open(os.path.join(self.path, self.__DESCR_FILENAME), 'w') as f:
            print("Writting :\n{}".format(self))
            f.write(self.__str__())

    def add_entry(self, notebook_name, row):
        self.notebooks[notebook_name].add_entry(row)

    def increase_entry_number(self):
        self.entry_number += 1
        return self.entry_number

    def save_image(self, image, filename='', extension=None):
        if extension == None:
            extension = self.image_format
        image.save(os.path.join(self.path_images,
                                "{}_{}.{}".format(filename, self.entry_number,
                                                  extension)))

    # TODO add support to matplotlib.pyplot.figure or add an additional
    # function
    def save_figure(self, fig, filename=None, extension=None):
        if extension == None:
            extension = self.fig_format
        fig.tight_layout()
        if filename is None:
            filename = fig.get_label()
        fig.savefig(os.path.join(self.path_figures,
                                "{}_{}.{}".format(filename, self.entry_number,
                                                  extension)))

    def __str__(self):
        return ("Date: {}\nName : {}\nPath : {}\nOverwrite : {}\n"
                "Image_format : {}\nFigure_format : {}"
                "").format(self.creation_date, self.name, self.path,
                        self.overwrite, self.image_format, self.fig_format)

    def get_shared_vars(self):
        s_vars = vars(self)
        s_vars['notebooks'] = {}
        return s_vars

    def set_shared_vars(self, s_vars):
        for key, value in s_vars.items():
            self.__dict__[key] = value

class SharedDiary(Diary):
    def __init__(self, s_vars, unique_id):
        self.set_shared_vars(s_vars)
        self.creation_date = datetime.datetime.now()
        self.uid = unique_id

        if self.stdout:
            self.redirect_stdout(self.path, 'stdout_uid_{}.txt'.format(self.uid))
        if self.stderr:
            self.redirect_stderr(self.path, 'stderr_uid_{}.txt'.format(self.uid))

    def add_notebook(self, name, **kwargs):
        name = '{}_uid_{}'.format(name, self.uid)
        return super(SharedDiary, self).add_notebook(name, **kwargs)


if __name__ == "__main__":
    diary = Diary(name='world', path='hello', overwrite=False)

    diary.add_notebook('validation')
    diary.add_notebook('test')

    diary.add_entry('validation', ['accuracy', 0.3])
    diary.add_entry('validation', ['accuracy', 0.5])
    diary.add_entry('validation', ['accuracy', 0.9])
    diary.add_entry('test', ['First test went wrong', 0.345, 'label_1'])

    image = Image.new(mode="1", size=(16,16), color=0)
    diary.save_image(image, filename='test_results')
