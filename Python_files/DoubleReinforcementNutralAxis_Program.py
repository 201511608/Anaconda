# 3
#Data # Double reinforce Concrete
#Inclued # Max Min condition :: for c-t to filder garbage data
import math
import numpy as np

file=open('DoubleReinforcementNutralAxis.csv','w+')

fckList=[20,25,30,40,50,60]
fstList=[415,500]
d_List=[10,20,30,40,50,60]


for fck in fckList:
    print(fck)
    for fst in fstList:  
        for d_ in d_List:
            for d in range(230,750+10,10):
                for b in range(230,750+10,10):
                    for Ast in range(300,3000+100,100):
                        for Asc in range(300,3000+100,100):
                            #1
                            fst=fst
                            if fst == 415:
                                strain=[0.00144,0.00163,0.00192,0.00241,0.00276,0.00380]
                                stress=[288,    306,    324   ,342    ,351     ,360]
                            elif fst ==500:
                                strain=[0.00174,0.00195,0.00226,0.00277,0.00312,0.00417]
                                stress=[347,    369,    391   ,413    ,423     ,434]

                            fck=fck
                            d_=d_
                            d=d
                            b=b
                            Ast=Ast 
                            Asc=Asc
                            x=0.00001  # Assume the Start of nutral axixnutral axis   # Assume 0 to d
                            F_diff=[]
                            minimum=9999999999999999999999999
                            nutral_axis=0.00001
                            Tension=0
                            Compression=0
                            while x<= d:
                                esc=0.0035*(1- d_/x)
                                fsc=np.interp(esc, strain, stress)

                                Cs=fsc*Asc
                                Cc=0.36*fck*b*x
                                C=Cc+Cs

                                # Tensile Steel Yields
                                est=0.0035*(d-x)/x
                                if est >=0.0035: 
                                    T=0.87*fst*Ast
                                else:
                                    T=fst*Ast



                                if abs(T-C)<=minimum:   # 
                                    minimum=abs(T-C)
                                    nutral_axis= x
                                    Tension=T
                                    Compression=C
                                    #break       
                                x=0.001+x 
                            ###
#                             print(fck, fst, b,d,d_,Ast,Asc,nutral_axis, Tension,Compression,Tension-Compression)
                            #Store All data later filter this data !_! 
                            file.write(str(fck)+", "+str(fst)+", "+str(b)+", "+str(d)+", "+str(d_)+", "+str(Ast)+", "+str(Asc)+", "+str(nutral_axis)+", "+str(Tension)+", "+str(Compression)+", "+str(Tension-Compression)+"\n")
file.close()