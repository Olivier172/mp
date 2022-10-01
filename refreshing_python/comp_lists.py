"""
Testing out comprehension lists in python
SYNTAX of a comprehension list:
newlist = [expression for item in iterable if condition == True]
<<<The return value is a new list, leaving the old list unchanged.>>>
"""

def main():
    print("Comprehension lists:")

    #numerical test
    list_with_values = range(0,20)
    ls1 = [x for x in list_with_values if x>=1] #this filters the list_with_values to keep only what you want in ls
    print(ls1)

    #alphabetic test
    list_with_letters="abcdefghijklmnoaaaa"
    ls2=[ltr for ltr in list_with_letters if ltr not in "abcd"] #filter out the letters abcd
    print(ls2)

    #filtering words
    list_with_words=["just","trying","this","out"]
    ls3=[wrd for wrd in list_with_words if wrd != "out"]
    print(ls3)

    #only words in filter set
    list_with_other_words=["monday","tuesday","wednesday","thursday","friday","saterday","sunday"]
    filter=["monday","wednesday","friday"]
    ls4=[wrd for wrd in list_with_other_words if wrd in filter]
    print(ls4)

    #creating a list with a lot of numbers fast
    ls5=[x for x in range(0,100)]
    print(ls5)  



#calling the main function
main()