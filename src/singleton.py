'''
Defines a Singleton generic class to be inherited by concrete singleton classes accross the project.
'''
class Singleton:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Singleton, cls).__new__(cls)
        return cls.instance