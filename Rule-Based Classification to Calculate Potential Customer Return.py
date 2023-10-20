# Türkiye’den Android kullanıcısı olan 33 yaşındaki bir kadının kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.

# Kural Tabanlı Sınıflandırma

import seaborn as sns
import pandas as pd
import numpy as np

# GÖREV 1

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz

pd.set_option("display.max_rows", None)
df = pd.read_csv('persona.csv')
df.head()
df.info()
df.shape

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?

df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# Soru 3: Kaç unique PRICE vardır?

df["PRICE"].nunique()

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

df["PRICE"].value_counts()

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?

df["COUNTRY"].value_counts()

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

df.groupby("COUNTRY").agg({"PRICE": "sum"})

# Soru 7: SOURCE türlerine göre satış sayıları nedir?

df["SOURCE"].value_counts()

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?

df.groupby("COUNTRY").agg({"PRICE": "mean"})

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?

df.groupby("SOURCE").agg({"PRICE": "mean"})

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?

df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})

# GÖREV 2

#  COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?

df.groupby(["COUNTRY", "SOURCE", "SEX","AGE"]).agg({"PRICE": "mean"})

# Görev 3: Çıktıyı PRICE’a göre sıralayınız.
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE’a göre uygulayınız.
# • Çıktıyı agg_df olarak kaydediniz.

df.groupby(["COUNTRY", "SOURCE", "SEX","AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)

agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX","AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)

# Görev 4: Indekste yer alan isimleri değişken ismine çeviriniz
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir. Bu isimleri değişken isimlerine çeviriniz.

agg_df = agg_df.reset_index()

# Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
# • Age sayısal değişkenini kategorik değişkene çeviriniz.
# • Aralıkları ikna edici şekilde oluşturunuz.
# • Örneğin: ‘0_18', ‘19_23', '24_30', '31_40', '41_70'

# Age değişkeninin bölüneceği aralıklar:

agg_df.head()

bins = [0,18,23,30,40, agg_df["AGE"].max()]

# Bölünen noktalara karşılık isimlendirmelerin ne olacağını ifade edelim:
mylabel = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]

# age'i böle işlemi:
agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins, labels=mylabel)
agg_df.head()

# Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız. • Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.
# • Yeni eklenecek değişkenin adı: customers_level_based
# • Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based değişkenini oluşturmanız gerekmektedir

agg_df["customers_level_based"] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'age_cat']].agg(lambda x: '_'.join(x).upper(), axis=1)

agg_df.head()

# customers_level_based değişkeninin her birinin tekilleşmesini istiyoruz fakat kontrol sonucu birden fazla mevcut
agg_df["customers_level_based"].value_counts()

# Bu sebeple segmentlere göre groupby yaptıktan sonra price ortalamalarını almalı ve segmentleri tekilleştirmeliyiz.

agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})
agg_df.head()

# customers_level_based değişkeni index olarak kaldı. değiştiriyoruz.

agg_df = agg_df.reset_index()
agg_df.head()

# her segmentten bir adet olmalı. kontrolünü sağlıyoruz.

agg_df["customers_level_based"].value_counts()
agg_df.head()

# GÖREV 7: Yeni müşterileri (USA_ANDROID_MALE_0_18) segmentlere ayırınız.
#############################################
# PRICE'a göre segmentlere ayırınız,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz, Segmentlere göre group by yapıp price mean, max, sum’larını alınız).

pd.qcut(agg_df["PRICE"], 4, labels= ["D", "C", "B", "A"])

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels= ["D", "C", "B", "A"])
agg_df.head()

agg_df.groupby("SEGMENT").agg({"PRICE": "mean"})
agg_df.groupby("SEGMENT").agg({"PRICE": "max"})
agg_df.groupby("SEGMENT").agg({"PRICE": "sum"})

agg_df.head(30)

# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.
# • 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
# • 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?

new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]

new_user_2 = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user_2]
