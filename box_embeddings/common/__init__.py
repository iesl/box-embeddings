ALLENNLP_PRESENT = True
try:
    import allennlp
except ImportError as ie:
    ALLENNLP_PRESENT = False
