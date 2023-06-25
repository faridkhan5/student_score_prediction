from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this func will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')
        
        return requirements


 
setup(
    name='mlproject',
    version='0.0.1',
    #the version of your application
    author='Khan',
    author_email='faridk302@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    #to install the required libs
)