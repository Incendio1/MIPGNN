# MIPGNN
## Usage Example
### MIPGNN
Cora 
```javascript 
python main.py --dataset=Cora --epochs=3000 --early_stopping=0 --hidden=16 --lr=0.01 --wd1=0.006 --wd2=0.006 --dropout1=0.8 --dropout2=0.8 --K=10 --ild_layer=10 --setting=semi --shuffle=fix --agg=sum --layers 1 2 3 4
```
Actor
```javascript 
python main.py --dataset=Actor --epochs=500 --early_stopping=200 --hidden=32 --lr=0.01 --wd1=0.001 --wd2=0.001 --dropout1=0.5 --dropout2=0.5 --K=10 --ild_layer=10 --setting=full --shuffle=random --agg=sum --layers 8 10
```
### MIPGNNa
Cora 
```javascript 
python main.py --dataset=Cora --epochs=3000 --early_stopping=500 --hidden=32 --lr=0.01 --wd1=0.006 --wd2=0.007 --wd3=0.01 --dropout1=0.8 --dropout2=0.8 --K=10 --tree_layer=10 --setting=semi --shuffle=fix --agg=weighted_sum --layers 1 2 3 4
```
Actor
```javascript 
python main.py --dataset=Actor --epochs=500 --early_stopping=200 --hidden=32 --lr=0.01 --wd1=0.003 --wd2=0.003 --wd3=0.001 --dropout1=0.5 --dropout2=0.5 --K=10 --ild_layer=10 --setting=full --shuffle=random --agg=weighted_sum --layers 3 4
```
## Results
model	|Cora	|CiteSeer	|PubMed|Cornell|Texas	|Wisconsin	|Actor
------ | -----  |----------- |---|--- | -----  |----------- |
MIPGNN|	85.4% |	74.0%|	80.5%|91.2%|	92.9% |	91.6%|40.7%
------ | -----  |----------- |---|--- | -----  |----------- |
MIPGNNa|84.4% |72.3%|79.9%|91.4%|92.2%|92.9%|41.1%

model |Cora |CiteSeer |PubMed|Cornell |Amazon Photo |Coauthor CS
------ | -----  |----------- |---|--- | -----  |----------- |
NE-WNA| 82.8% | 74.2%| 82.5%|84.7%| 93.2% | 92.5%|


