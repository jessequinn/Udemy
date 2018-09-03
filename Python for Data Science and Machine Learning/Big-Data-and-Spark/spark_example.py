from pyspark import SparkContext

sc = SparkContext()

# write file in jupyter
# %%writefile example.txt
# first line
# second line
# third line
# fourth line

# RDD
textFile = sc.textFile('./example.txt')

# actions
# print(textFile.count())
# print(textFile.first())

# transformation
secfind = textFile.filter(lambda line: 'second' in line)
# print(secfind)
# print(secfind.collect())
# print(secfind.count())

# RDD
textFile2 = sc.textFile('./example2.txt')
words = textFile2.map(lambda line: line.split())

# print(words.collect())

# print(textFile2.flatMap(lambda line: line.split()).collect())

textFile3 = sc.textFile('./services.txt')

# print(textFile3.take(2))
# print(textFile3.map(lambda line: line.split()).take(3))

# print(textFile3.map(lambda line: line[1:]
                    # if line[0] == '#' else line).collect())
clean = textFile3.map(lambda line: line[1:] if line[0] == '#' else line)
clean = clean.map(lambda line: line.split())
# print(clean.collect())

# print(clean.map(lambda line: (line[3], line[-1])).collect())
pairs = clean.map(lambda line: (line[3], line[-1]))
rekey = pairs.reduceByKey(lambda amt1, amt2: float(amt1) + float(amt2))
# print(rekey.collect())

# step 1 Grab (state, amount)
step1 = clean.map(lambda lst: (lst[3], lst[-1]))

# step 2 reduce by key
step2 = step1.reduceByKey(lambda amt1, amt2: float(amt1) + float(amt2))

# step 3 remove state amount titles
step3 = step2.filter(lambda x: not x[0] == 'State')

# step 4 sort results by amount
step4 = step3.sortBy(lambda stAmount: stAmount[1], ascending=False)

print(step4.collect())

x = ['ID', 'State', 'Amount']

def func1(lst):
    return lst[-1]

# tuple unpacking
def func2(id_st_amt):
    # unpack values
    (id, st, amt) = id_st_amt
    return amt

print(func1(x))
print(func2(x))