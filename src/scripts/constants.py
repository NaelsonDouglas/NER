KEEP_COLUMNS = [
    TITLE := 'title',
    MAKE := 'make',
    MODELNAME := 'modelname',
    MODELNAMEQ := 'modelnameq',
    MODELNO := 'modelno',
    MODELNOQ := 'modelnoq',
]
FEATURES = [MAKE, MODELNAME,MODELNAMEQ, MODELNO, MODELNOQ]

MIN_FUZZY_RATIO = 85

_NORMALIZED_PREFIX = '_SUGGESTION'
MAKE_NORMALIZED = MAKE.upper()+_NORMALIZED_PREFIX
MODELNAME_NORMALIZED = MODELNAME.upper()+_NORMALIZED_PREFIX
MODELNO_NORMALIZED = MODELNO.upper()+_NORMALIZED_PREFIX

LOCAL_STOPWORDS = set(['sale','sell', 'loan','buy','lease','auction', 'available', 'service', 'selling'])