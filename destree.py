class descisionnode:
        def __init__(self,col=-1,value = None , results = None , tb = None , fb = None):
                self.col = col
                self.value = value
                self.results = results
                self.fb = fb
                self.tb = tb

def divideset(rows,column,value):
        split_function = None      
        if isinstance(value,int) or isinstance(value,float):
                split_function = lambda row:row[column]>=value  
        else:
                split_function = lambda row:row[column]==value

        set1 = [row for row in rows if split_function(row)]
        set2 = [row for row in rows if not split_function(row)]
        return (set1,set2)

def uniquecounts(rows):
        results= {}
        for row in rows:
                r = row[len(row)-1]
                if r not in results: 
                        results[r]=0
                results[r] += 1
        
        return results

def entropy(rows):
        from math import log
        log2 = lambda x:log(x)/log(2)
        results=uniquecounts(rows)

        ent = 0.0
        for r in results.keys():
                p = float(results[r])/len(rows)
                print p
                ent = ent - p*log2(p)
        return ent

my_data=[['slashdot','USA','yes',18,'None'],
        ['google','France','yes',23,'Premium'],
        ['reddit','USA','yes',24,'Basic'],
        ['kiwitobes','France','yes',23,'Basic'],
        ['google','UK','no',21,'Premium'],
        ['(direct)','New Zealand','no',12,'None'],
        ['(direct)','UK','no',21,'Basic'],
        ['google','USA','no',24,'Premium'],
        ['slashdot','France','yes',19,'None'],
        ['reddit','USA','no',18,'None'],
        ['google','UK','no',18,'None'],
        ['kiwitobes','UK','no',19,'None'],
        ['reddit','New Zealand','yes',12,'Basic'],
        ['slashdot','UK','no',21,'None'],
        ['google','UK','yes',18,'Basic'],
        ['kiwitobes','France','yes',19,'Basic']]

print uniquecounts(my_data)
print len(my_data)
print entropy(my_data)