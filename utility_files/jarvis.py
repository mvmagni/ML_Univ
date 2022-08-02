# -*- coding: utf-8 -*-

from datetime import datetime
import pytz
import os
from os import listdir
from os.path import isfile, join
import pathlib
import cw_file_utils as cw_fu
import pkg_resources
import pickle
import gzip

class Jarvis:
  # lowercase are internal to class params
  timezone = 'America/New_York'
  startuptime = datetime.now(pytz.timezone(timezone))
  longDateFormat = "%d-%b-%Y, %H:%M"
  shortTimeFormat = "%H:%M"
  utility_files_dir = 'utility_files'
  data_dir = 'data'
  data_subdir1 = '01_original'
  data_subdir2 = '02_working'
  data_subdir3 = '03_train'
  data_subdir4 = '04_test'
  data_subdir5 = '05_experiments'

  data_subdirs = [data_subdir1, data_subdir2, data_subdir3, data_subdir4, data_subdir5]

  PROJECT_NAME = ''
  ROOT_DIR = ''
  ROOT_DATA_DIR = ''
  DATA_DIR = ''
  DATA_DIR_ORIG = ''
  DATA_DIR_WORK = ''
  DATA_DIR_TEST = ''
  DATA_DIR_TRAIN = ''
  WORKING_DIR = ''
  UTILITY_DIR = ''


  def __init__(self,
               ROOT_DIR,
               PROJECT_NAME
               ):

    self.ROOT_DIR = ROOT_DIR
    self.PROJECT_NAME = PROJECT_NAME
    self.UTILITY_DIR = join(self.ROOT_DIR,self.utility_files_dir)

    self.startup()
    #self.showenvironment()
    self.environmentScan()
    self.greeting()

  def startup(self):
    print("Wha...where am I?")
    self.ROOT_DATA_DIR = join(self.ROOT_DIR, self.data_dir)
    self.DATA_DIR = join(self.ROOT_DATA_DIR, self.PROJECT_NAME)
    self.WORKING_DIR = join(self.ROOT_DIR, self.PROJECT_NAME)
    self.DATA_DIR_ORIG = join(self.DATA_DIR, self.data_subdir1)
    self.DATA_DIR_WORK = join(self.DATA_DIR, self.data_subdir2)
    self.DATA_DIR_TRAIN = join(self.DATA_DIR, self.data_subdir3)
    self.DATA_DIR_TEST = join(self.DATA_DIR, self.data_subdir4)
    self.DATA_DIR_EXP = join(self.DATA_DIR, self.data_subdir5)
    print("I am awake now.")

  def whattimeisit(self):
      now = datetime.now(pytz.timezone(self.timezone))
      print("The current time is " + now.strftime(self.shortTimeFormat))

  def showenvironment(self):
      print("I am inspecting the local environment...")
      print('')
      print("Your environment has been configured: ")
      print("PROJECT_NAME:     " + self.PROJECT_NAME)
      print("ROOT_DIR: " + self.ROOT_DIR)
      print("WORKING_DIR:  " + self.WORKING_DIR)
      print('')
      print("ROOT_DATA_DIR: " + self.ROOT_DIR)
      print("DATA_DIR:     " + self.DATA_DIR)
      print("DATA_DIR_ORIG:" + self.DATA_DIR_ORIG)
      print("DATA_DIR_WORK:" + self.DATA_DIR_WORK)
      print("DATA_DIR_TRAIN:" + self.DATA_DIR_TRAIN)
      print("DATA_DIR_TEST:" + self.DATA_DIR_TEST)
      print('')
      print("UTILITY_DIR:  " + self.UTILITY_DIR)

  def displayProjects(self):
      onlyDirs = [f for f in listdir(self.ROOT_DIR) if not isfile(join(self.ROOT_DIR, f))]
      onlyDirs.sort()
      print("Project listing:")
      for x in onlyDirs:
          print("--> " + x)

  def environmentScan(self):
      self.setupWorkingDir()
      self.setupDataDir()

      #self.showProjectWorkFiles()
      #self.showProjectDataFiles()

      # Set directory to WORKING_DIR
      os.chdir(self.WORKING_DIR)
      print("I have set your current working directory to {0}".format(os.getcwd()))

  def setupWorkingDir(self):
      if not os.path.exists(self.WORKING_DIR):
          # Create a new directory because it does not exist
          os.makedirs(self.WORKING_DIR)
          print("The working directory is not present. Directory created.")

  def setupDataDir(self):
      if not os.path.exists(self.DATA_DIR):
          os.makedirs(self.DATA_DIR)
          print("The project data directory is not present. Directory created.")

      for x in self.data_subdirs:
          if not(os.path.exists(join(self.DATA_DIR, x))):
              os.makedirs(join(self.DATA_DIR, x))
              print("Data subdirectory " + x + " has been created")

      print('')

  def greeting(self):
    greeting = ''
    x = int(self.startuptime.strftime("%H"))

    if (x > 7) and (x <= 8):
      greeting = 'An early morning I see.'
    elif (x > 8) and (x <= 12):
      greeting = 'Extra caffeine may help.'
    elif (x > 12) and (x <= 18):
      greeting = 'Reminder, no more coffee.'
    elif (x > 18) and (x <= 20):
      greeting = 'I hope you had dinner.'
    elif (x > 20) and (x <= 24):
      greeting = 'I see you are having a productive evening.'
    elif (x <= 7):
      greeting = 'You should really be sleeping.'

    self.whattimeisit()
    print("Hello sir. " + greeting)
    print('')

  def showProjectDataFiles(self):
      print("Here are all your project data files")
      cw_fu.exploreDirectory(self.DATA_DIR)

  def showProjectWorkFiles(self):
      print("Here are all your project work files")
      cw_fu.exploreDirectory(self.WORKING_DIR)

  def showAllDataFiles(self):
      print("Here are all your available data files")
      cw_fu.exploreDirectory(self.ROOT_DATA_DIR)

  def compressProjectDataFiles(self, removeOriginal=False):
      cw_fu.exploreDirectory(self.DATA_DIR, compress=True, removeOriginal=removeOriginal)

  def compressAllDataFiles(self, removeOriginal=False):
      cw_fu.exploreDirectory(self.ROOT_DATA_DIR, compress=True, removeOriginal=removeOriginal)

  def getPackageVersion(self, pkgName):
      print(pkgName + " version: " + str(pkg_resources.get_distribution(pkgName)))

  def saveExperiment(self, dataExperiment, fileName, fileExtension='.jexp'):
      pickle.dump(dataExperiment, gzip.open(f'{self.DATA_DIR_EXP}/{fileName}{fileExtension}.gz', 'wb'))

  def loadExperiment(self, fileName, fileExtension='.jexp'):
      obj = pickle.load(gzip.open(f'{self.DATA_DIR_EXP}/{fileName}{fileExtension}.gz', 'rb'))
      return obj
