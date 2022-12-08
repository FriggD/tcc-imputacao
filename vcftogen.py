import time
import pandas as pd

def readLines(file, lines: int):
    lineData = []
    for _ in range(lines):
        lineData.append(file.readline())
    return lineData
        
def getFileHandler(filename: str):
    f = open(filename, 'r')
    for _ in range(300):
        line = f.readline()
        if line.split("\t")[0].strip() == '#CHROM':
            return f
        
def callRateCheck(line):
    for idx, a in enumerate(line):
        gen = a.split("|")[0]
        if gen == '2':
            return False
    return True

def main():
    f = getFileHandler('chr1.vcf')
    lines = readLines(f, 20000)
    
    GEN = [ [] for _ in range(len(lines[0].split('\t')) - 8)]
    for line in lines:
        allelesArr = line.split('\t')[9:]
        index = 0
        
        if callRateCheck(allelesArr) == False:
            continue
        
        for idx, a in enumerate(allelesArr):
            gen = a.split("|")[0]
            GEN[index].append(gen)
            index +=1
    
    f = open("output.txt", 'w')    
    for g in GEN:
        time.sleep(1)
        f.write("".join(g)+ "\n")
        
        
if __name__ == '__main__':
    main()