def tej():
    v=['a','e','i','o','u']
    c=['b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','y','z']
    n=input("Enter the string: ").lower()
    c1=0
    c2=0
    for i in n:
        if i in v:
            c1=c1+1 #count for vowels
        elif i in c:
            c2=c2+1 # count for consonants
    return c1,c2
vc,cc=tej()
print("No. of vowels in the string are: ",vc)
print("No. of consonents in the string are: ",cc)
