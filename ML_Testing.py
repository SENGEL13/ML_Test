import streamlit as st
import pandas as pd
from io import StringIO
import altair as alt
from machine_learning_test.ML_Test import ML_Collective

source = pd.DataFrame({
    'a': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
    'b': [28, 55, 43, 91, 81, 53, 19, 87, 52]
})

def file_input():#文件导入数据
    uploaded_file = st.file_uploader("选择文件")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("gb18030"))
        # To read file as string:
        string_data = stringio.read()
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write('数据:')
        st.dataframe(dataframe)
        return dataframe

st.title("机器学习算法测试")
df = file_input()
df_result = None
if type(df) != type(None):
    data,label,factor,time,proportion = df,df.columns[-1],df.columns[0:-1],st.number_input("输入每个算法需要重复计算的次数:",step=1,help='建议设置为100',value=10),st.number_input("输入测试集占比:",step=0.1,min_value=0.1,max_value=0.9,value=0.4)
    if time==0:
        st.stop()
    df_result = ML_Collective(data,label,factor,time,proportion=proportion)
    df_result['综合评分'] = df_result['平均值']*0.4+(1-df_result['标准差'])*0.3+df_result['最大值']*0.2+df_result['最小值']*0.1
    st.write(f"每个算法{time}次计算，精确度比较(交叉验证 训练集:测试={1-proportion}:{proportion}):")
    st.write(df_result)
    sorted_max = df_result.sort_values(by='最大值',ascending=False)[['算法','最大值']]
    name_max = list(sorted_max['算法'])

    sorted_min = df_result.sort_values(by='最小值',ascending=False)[['算法','最小值']]
    name_min = list(sorted_min['算法'])

    sorted_arr = df_result.sort_values(by='平均值',ascending=False)[['算法','平均值']]
    name_arr = list(sorted_arr['算法'])

    sorted_std = df_result.sort_values(by='标准差',ascending=True)[['算法','标准差']]
    name_std = list(sorted_std['算法'])

    sorted_all = df_result.sort_values(by='综合评分',ascending=False)[['算法','综合评分']]
    name_all = list(sorted_all['算法'])
    #st.write(ahead_max,ahead_min,ahead_arr,ahead_std)

    with st.container():
        """# $\color{#362B48}{分析报告}$"""
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
"""## 根据准确的最大值推荐\n
># $\color{#008CFF}"""+"""{:}$""".format(name_max[0])+"""\n
>## {:}\n
> {:}\n
> {:}\n
> {:}\n""".format(name_max[1],name_max[2],name_max[3],name_max[4]))
        with col2:
            st.subheader('准确度排行条形图')
            c = alt.Chart(sorted_max).mark_bar().encode(
                x='最大值',
                y=alt.Y('算法', sort='-x')
            )
            st.altair_chart(c)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown(
"""## 根据准确的最小值推荐\n
># $\color{#008CFF}"""+"""{:}$""".format(name_min[0])+"""\n
>## {:}\n
> {:}\n
> {:}\n
> {:}\n""".format(name_min[1],name_min[2],name_min[3],name_min[4]))
        with col4:
            st.subheader('准确度排行条形图')
            c = alt.Chart(sorted_min).mark_bar().encode(
                x='最小值',
                y=alt.Y('算法', sort='-x')
            )
            st.altair_chart(c)

        col5, col6 = st.columns(2)
        with col5:
            st.markdown(
"""## 根据准确的平均值推荐\n
># $\color{#008CFF}"""+"""{:}$""".format(name_arr[0])+"""\n
>## {:}\n
> {:}\n
> {:}\n
> {:}\n""".format(name_arr[1],name_arr[2],name_arr[3],name_arr[4]))
        with col6:
            st.subheader('准确度排行条形图')
            c = alt.Chart(sorted_arr).mark_bar().encode(
                x='平均值',
                y=alt.Y('算法', sort='-x')
            )
            st.altair_chart(c)

        col7, col8 = st.columns(2)
        with col7:
            st.markdown(
"""## 根据准确的标准差推荐\n
># $\color{#008CFF}"""+"""{:}$""".format(name_std[0])+"""\n
>## {:}\n
> {:}\n
> {:}\n
> {:}\n""".format(name_std[1],name_std[2],name_std[3],name_std[4]))
        with col8:
            st.subheader('准确度排行条形图')
            c = alt.Chart(sorted_std).mark_bar().encode(
                x='标准差',
                y=alt.Y('算法', sort='x')
            )
            st.altair_chart(c)

        col9, col10 = st.columns(2)
        with col9:
            st.markdown(
"""## 根据准确的综合评分推荐\n
$P = \overline{S} \\times 0.4 +(1-\sigma) \\times0.3+x_{max}\\times 0.2+x_{min}\\times 0.1$\n
># $\color{#008CFF}"""+"""{:}$""".format(name_all[0])+"""\n
>## {:}\n
> {:}\n
> {:}\n
> {:}\n""".format(name_all[1],name_all[2],name_all[3],name_all[4]))
        with col10:
            st.subheader('准确度排行条形图')
            c = alt.Chart(sorted_all).mark_bar().encode(
                x='综合评分',
                y=alt.Y('算法', sort='-x')
            )
            st.altair_chart(c)
        
