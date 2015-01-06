import sqlalchemy
from . import analyze



class _Disapprint(object):
    """
    Handles numerical processing and database interaction
    """

    def __init__(self, *args, **kwargs):
        pass

    def add_fingerprints(self, fp, format = None):
        fingerprint_hashes = analyze.get_fingerprints(fp, format)
        return fingerprint_hashes


class Disapprint(_Disapprint):
    """
    User facing API
    """

    def __init__(self, *args, **kwargs):
        super(Disapprint, self).__init__(args, kwargs)

        if 'db' in kwargs:
            self.db_filepath = kwargs['db']
        else:
            self.db_filepath = './disapprint.db'

    def new_source(self, fp):
        foo = self.add_fingerprints(fp)
        return foo
        
    def analyze_sample(self, fp):
        pass