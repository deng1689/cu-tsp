Brief instruction for running TM-score program:
(For detail: Zhang & Skolnick,  Proteins, 2004 57:702-10)
                                                          
1. Run TM-score to compare 'model' and 'native:          
   >java -jar TMscore.jar model native                   
                                                         
2. Run TM-score with an assigned d0, e.g. 5 Angstroms:   
   >java -jar TMscore.jar model native -d 5              
                                                         
3. Run TM-score with superposition output, e.g. 'TM.sup':
   >java -jar TMscore.jar model native -o TM.sup         
   To view the superimposed structures by rasmol:        
   >rasmol -script TM.sup                                

