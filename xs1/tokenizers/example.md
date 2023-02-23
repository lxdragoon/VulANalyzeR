
```python

from tokenizer import blk2seq
import sentencepiece as spm


model = 'tokenizer_10000000_30000.sp.model'

sp = spm.SentencePieceProcessor()
sp.Load(model)

ids = blk2seq(a_block, sp, flatten=False)
print(ids)


```

example process:

```
original:
['mov edi [ eax + 20h ]', 'cmp edi ebx', 'mov [ ebp + var _ 68 ] ebx', 'jz loc _ 414713']

tokenized into word pieces by sp:
[['▁mov', '▁edi', '▁', '[', '▁', 'eax', '▁', '+', '▁20', 'h', '▁]'], ['▁cmp', '▁edi', '▁', 'ebx'], ['▁mov', '▁', '[', '▁', 'ebp', '▁', '+', '▁var', '▁', '_', '▁68', '▁]', '▁', 'ebx'], ['▁j', 'z', '▁loc', '▁', '_', '▁41471', '3']]

mapped into ids:
[[7, 23, 5, 9, 5, 8, 5, 11, 82, 17, 10], [47, 23, 5, 19], [7, 5, 9, 5, 13, 5, 11, 20, 5, 6, 160, 10, 5, 19], [34, 31, 24, 5, 6, 6752, 65]]


```


```
to get vocab sizeL

sp.vocab_size()
```

