import os

def getPath():
    # return os.getcwd()
    return "/home/mingfei/Documents/projects/RobotManipulationProject/src/gail/scripts"

def getDataPath():
    # return os.getcwd()
    return "/home/mingfei/Documents/projects/RobotManipulationProject/src/gail/scripts/data"

def get_task_name(args):
    task_name = 'ee_lfd'
    task_name += ".seed_{}".format(args.seed)
    return task_name