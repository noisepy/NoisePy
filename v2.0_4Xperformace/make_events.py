from datetime import datetime

years = [2010,2011,2012]
for ii in range(len(years)):
    year = years[ii]

    for jj in range(1,13):
        month = jj

        if jj==1 or jj==3 or jj==5 or jj==7 or jj==9 or jj==10 or jj==12:
            iday = 31
        elif jj==2:
            iday = 28
        else:
            iday = 30
        
        for kk in range(1,iday+1):
            print('%04d_%02d_%02d' % (year,month,kk))