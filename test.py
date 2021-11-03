import pandas as pd


a = [3, 4, 5, 8943, 3429, 7639, 6540, 36435, 3054, 1917, 1975, 2574, 5383, 5286, 1889, 2006, 3400, 3720, 1910, 2493, 1917, 507, 63, 21, 65, 2349, 11745, 4971, 1900, 6346, 1901, 4666, 8826, 2315, 1917, 16, 3195, 1885, 3188, 2141, 2349, 2980, 17, 2897, 2315, 1917, 4233, 3195, 1885, 4830, 5383, 1920, 2454, 1917, 507, 15809, 10091, 1885, 1981, 1889, 10141, 1942, 1925, 2244, 1886, 507, 1909, 8943, 3356, 2255, 12853, 2889, 1885, 35109, 2036, 1910, 2244, 1886, 507, 1940, 9269, 2829, 2053, 13615, 38969, 14019, 1889, 2058, 2244, 1886, 1900, 2143, 1945, 1909, 5383, 6411, 2016, 13510, 1885, 4982, 1889, 3660, 1886, 1994, 3054, 1898, 5286, 1904, 5058, 1910, 18979, 1910, 2953, 1886, 1900, 3054, 7185, 1920, 1993, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
each_len = 50
tmp =int(len(a) / each_len) 
print(len(a))
print(tmp)
for i in range(tmp):
    print(a[each_len * i : each_len * (i + 1)])

print(a[each_len * (i + 1):])

print(len([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))

def main(): 
    tamil_data, hindi_data = data_slice()
    tamil_data_question = tamil_data[['question']]
    hindi_data_question = hindi_data[['question']]
   
    print('end')
    print(list(tamil_data['context'])[0][53:167])

def data_slice():
    df = pd.read_csv('./train.csv')
    df = df[['id','context','question','answer_text','language']]
    
    df1 = df[df['language'].apply(lambda language: 'tamil' == language)]
    df2 = df[df['language'].apply(lambda language: 'hindi' == language)]

    #reset index of df1 and df2 
    df1 = df1[['id','context','question','answer_text','language']].reset_index(drop=True)
    df2 = df2[['id','context','question','answer_text','language']].reset_index(drop=True)

    #print(df1)
    #print('===================')
    #print(df2)
    return df1, df2 

if __name__ == "__main__":
    main()

