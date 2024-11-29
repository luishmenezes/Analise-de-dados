import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\luisf\\OneDrive\\Área de Trabalho\\faculdade\\Residencia\\BI\\SPOILER_Saúde - Base de Atendimentos (1).csv")

print(df.head())
print(df.info())
print(df.nunique())

plt.figure(figsize=(10, 6))
df['Idade'].hist(bins=20, color='skyblue')
plt.title('Distribuição de Idade dos Pacientes', fontsize=16)
plt.xlabel('Idade', fontsize=14)
plt.ylabel('Frequência', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
df['Gênero'].value_counts().plot(kind='bar', color=['blue', 'pink'])
plt.title('Distribuição por Gênero', fontsize=16)
plt.ylabel('Contagem', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
df['Estado Civil'].value_counts().plot(kind='bar', color='green')
plt.title('Distribuição por Estado Civil', fontsize=16)
plt.ylabel('Contagem', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
df.groupby('Especialidade')['Valor Consulta'].mean().sort_values().plot(kind='barh', color='orange')
plt.title('Valor Médio de Consulta por Especialidade', fontsize=16)
plt.xlabel('Valor Médio', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
df['Primeira Consulta ou Retorno'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen'])
plt.title('Distribuição de Consultas: Primeira vs. Retorno', fontsize=16)
plt.ylabel('')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
df['Diagnóstico'].value_counts().head(10).plot(kind='bar', color='purple')
plt.title('Top 10 Diagnósticos mais Frequentes', fontsize=16)
plt.ylabel('Contagem', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
df.groupby('Especialidade')['Exame Necessário'].value_counts().unstack().plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Exames Necessários por Especialidade', fontsize=16)
plt.ylabel('Contagem', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Idade', y='Valor Consulta', data=df, hue='Gênero', alpha=0.7)
plt.title('Relação entre Idade e Valor da Consulta', fontsize=16)
plt.xlabel('Idade', fontsize=14)
plt.ylabel('Valor da Consulta', fontsize=14)
plt.tight_layout()
plt.show()

df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
if df['Data'].isnull().any():
    print("Foram encontrados valores inválidos na coluna 'Data':")
    print(df[df['Data'].isnull()])

df['Ano'] = df['Data'].dt.year
df['Mês'] = df['Data'].dt.month

consultas_tempo = df.groupby(['Ano', 'Mês']).size().reset_index(name='Consultas')

plt.figure(figsize=(12, 6))
plt.plot(consultas_tempo['Mês'].astype(str) + '/' + consultas_tempo['Ano'].astype(str),
         consultas_tempo['Consultas'], marker='o', color='blue')
plt.title('Número de Consultas ao Longo do Tempo', fontsize=16)
plt.xlabel('Mês/Ano', fontsize=14)
plt.ylabel('Número de Consultas', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
receita_por_medico = df.groupby('Médico')['Valor Consulta'].sum().sort_values(ascending=False)
receita_por_medico.head(10).plot(kind='bar', color='teal')
plt.title('Top 10 Médicos com Maior Receita', fontsize=16)
plt.ylabel('Receita Total', fontsize=14)
plt.xlabel('Médico', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df['Hora'] = pd.to_datetime(df['Data'], errors='coerce').dt.hour
consultas_por_horario = df['Hora'].value_counts().sort_index()
print("Distribuição de Consultas por Horário:")
print(consultas_por_horario)

df['Faixa Etária'] = pd.cut(df['Idade'], bins=[0, 18, 35, 50, 65, 100], labels=['0-18', '19-35', '36-50', '51-65', '65+'])
faixa_diagnostico = df.groupby(['Faixa Etária', 'Diagnóstico']).size().unstack(fill_value=0)
print("Diagnósticos por Faixa Etária:")
print(faixa_diagnostico)

receita_genero_faixa = df.groupby(['Gênero', 'Faixa Etária'])['Valor Consulta'].sum()
print("Receita por Gênero e Faixa Etária:")
print(receita_genero_faixa)

exames_receita = df.groupby(['Especialidade', 'Exame Necessário'])['Valor Consulta'].sum().unstack(fill_value=0)
print("Exames por Especialidade e Receita:")
print(exames_receita)

df_sorted = df.sort_values(by=['Paciente', 'Data'])
df_sorted['Intervalo Retorno'] = df_sorted.groupby('Paciente')['Data'].diff().dt.days
print("Tempo Médio de Retorno (em dias):", df_sorted['Intervalo Retorno'].mean())

receita_diagnostico_medico = df.groupby(['Médico', 'Diagnóstico'])['Valor Consulta'].sum().unstack(fill_value=0)
print("Receita por Diagnóstico e Médico:")
print(receita_diagnostico_medico)

retorno_diagnostico = df[df['Primeira Consulta ou Retorno'] == 'Retorno'].groupby('Diagnóstico').size()
print("Taxa de Retorno por Diagnóstico:")
print(retorno_diagnostico)

correlacao = df[['Idade', 'Valor Consulta']].corr()
print("Correlação entre Idade e Valor da Consulta:")
print(correlacao)

especialidade_faixa = df.groupby(['Faixa Etária', 'Especialidade']).size().unstack(fill_value=0)
print("Especialidades por Faixa Etária:")
print(especialidade_faixa)

consultas_paciente = df['Paciente'].value_counts()
print("Top 10 Pacientes com Consultas Frequentes:")
print(consultas_paciente.head(10))
