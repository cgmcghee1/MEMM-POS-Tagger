# MEMM-POS-Tagger

This is a POS tagger based on a Maximum Entropy Markov Model. The functions file contains some key functions for feature extraction and regression. Usage would be the following:

<p> >>> import MEMM </p>
<p> >>> a = MEMM.Model().train() </p>
<p> >>> a.tagger('Where is the train station?') </p>
['RB', 'VBZ', 'DT', 'NN', 'NN', '.']
