unloadNamespace("ggplot2")
install.packages("ggplot2")
library(ggplot2)
diabetes_data <- read.csv("D:\\dataset\\diabetes2.csv")
# График Pregnancies, Age, Outcome
ggplot(diabetes_data, aes(x = Pregnancies, y = Age, color = factor(Outcome))) +
  geom_point() +
  labs(title = "Scatter Plot: Pregnancies, Age, Outcome")

# График Glucose, Age, Outcome
ggplot(diabetes_data, aes(x = Glucose, y = Age, color = factor(Outcome))) +
  geom_point() +
  labs(title = "Scatter Plot: Glucose, Age, Outcome")

# График BloodPressure, Age, Outcome
ggplot(diabetes_data, aes(x = BloodPressure, y = Age, color = factor(Outcome))) +
  geom_point() +
  labs(title = "Scatter Plot: BloodPressure, Age, Outcome")

# График SkinThickness, Age, Outcome
ggplot(diabetes_data, aes(x = SkinThickness, y = Age, color = factor(Outcome))) +
  geom_point() +
  labs(title = "Scatter Plot: SkinThickness, Age, Outcome")

# График Insulin, Age, Outcome
ggplot(diabetes_data, aes(x = Insulin, y = Age, color = factor(Outcome))) +
  geom_point() +
  labs(title = "Scatter Plot: Insulin, Age, Outcome")

# График BMI, Age, Outcome
ggplot(diabetes_data, aes(x = BMI, y = Age, color = factor(Outcome))) +
  geom_point() +
  labs(title = "Scatter Plot: BMI, Age, Outcome")

# График DiabetesPedigreeFunction, Age, Outcome
ggplot(diabetes_data, aes(x = DiabetesPedigreeFunction, y = Age, color = factor(Outcome))) +
  geom_point() +
  labs(title = "Scatter Plot: DiabetesPedigreeFunction, Age, Outcome")

# Построение многовариантной гистограммы
ggplot(diabetes_data, aes(x = Pregnancies, y = Age, fill = factor(Outcome))) +
  geom_bin2d(bins = 20, color = "white") +
  labs(title = "Multivariate Histogram of Pregnancies and Age by Outcome", x = "Pregnancies", y = "Age", fill = "Outcome") +
  theme_minimal()

# Построение многовариантной коробчатой диаграммы
ggplot(diabetes_data, aes(x = factor(Outcome), y = Age, fill = factor(Outcome))) +
  geom_boxplot() +
  labs(title = "Multivariate Boxplot of Age by Outcome", x = "Outcome", y = "Age", fill = "Outcome") +
  theme_minimal()

# Построение многовариантной гистограммы
ggplot(diabetes_data, aes(x = Glucose, y = Age, fill = factor(Outcome))) +
  geom_bin2d(bins = 20, color = "white") +
  labs(title = "Multivariate Histogram of Glucose and Age by Outcome", x = "Glucose", y = "Age", fill = "Outcome") +
  theme_minimal()

# Построение многовариантной коробчатой диаграммы
ggplot(diabetes_data, aes(x = factor(Outcome), y = Age, fill = factor(Outcome))) +
  geom_boxplot() +
  labs(title = "Multivariate Boxplot of Age by Outcome", x = "Outcome", y = "Age", fill = "Outcome") +
  theme_minimal()

# Построение многовариантной гистограммы
ggplot(diabetes_data, aes(x = BloodPressure, y = Age, fill = factor(Outcome))) +
  geom_bin2d(bins = 20, color = "white") +
  labs(title = "Multivariate Histogram of BloodPressure and Age by Outcome", x = "BloodPressure", y = "Age", fill = "Outcome") +
  theme_minimal()

# Построение многовариантной коробчатой диаграммы
ggplot(diabetes_data, aes(x = factor(Outcome), y = Age, fill = factor(Outcome))) +
  geom_boxplot() +
  labs(title = "Multivariate Boxplot of Age by Outcome", x = "Outcome", y = "Age", fill = "Outcome") +
  theme_minimal()

# Построение многовариантной гистограммы
ggplot(diabetes_data, aes(x = SkinThickness, y = Age, fill = factor(Outcome))) +
  geom_bin2d(bins = 20, color = "white") +
  labs(title = "Multivariate Histogram of SkinThickness and Age by Outcome", x = "SkinThickness", y = "Age", fill = "Outcome") +
  theme_minimal()

# Построение многовариантной коробчатой диаграммы
ggplot(diabetes_data, aes(x = factor(Outcome), y = Age, fill = factor(Outcome))) +
  geom_boxplot() +
  labs(title = "Multivariate Boxplot of Age by Outcome", x = "Outcome", y = "Age", fill = "Outcome") +
  theme_minimal()

# Построение многовариантной гистограммы
ggplot(diabetes_data, aes(x = Insulin, y = Age, fill = factor(Outcome))) +
  geom_bin2d(bins = 20, color = "white") +
  labs(title = "Multivariate Histogram of Insulin and Age by Outcome", x = "Insulin", y = "Age", fill = "Outcome") +
  theme_minimal()

# Построение многовариантной коробчатой диаграммы
ggplot(diabetes_data, aes(x = factor(Outcome), y = Age, fill = factor(Outcome))) +
  geom_boxplot() +
  labs(title = "Multivariate Boxplot of Age by Outcome", x = "Outcome", y = "Age", fill = "Outcome") +
  theme_minimal()

# Построение многовариантной гистограммы
ggplot(diabetes_data, aes(x = BMI, y = Age, fill = factor(Outcome))) +
  geom_bin2d(bins = 20, color = "white") +
  labs(title = "Multivariate Histogram of BMI and Age by Outcome", x = "BMI", y = "Age", fill = "Outcome") +
  theme_minimal()

# Построение многовариантной коробчатой диаграммы
ggplot(diabetes_data, aes(x = factor(Outcome), y = Age, fill = factor(Outcome))) +
  geom_boxplot() +
  labs(title = "Multivariate Boxplot of Age by Outcome", x = "Outcome", y = "Age", fill = "Outcome") +
  theme_minimal()

# Построение многовариантной гистограммы
ggplot(diabetes_data, aes(x = DiabetesPedigreeFunction, y = Age, fill = factor(Outcome))) +
  geom_bin2d(bins = 20, color = "white") +
  labs(title = "Multivariate Histogram of DiabetesPedigreeFunction and Age by Outcome", x = "DiabetesPedigreeFunction", y = "Age", fill = "Outcome") +
  theme_minimal()

# Построение многовариантной коробчатой диаграммы
ggplot(diabetes_data, aes(x = factor(Outcome), y = Age, fill = factor(Outcome))) +
  geom_boxplot() +
  labs(title = "Multivariate Boxplot of Age by Outcome", x = "Outcome", y = "Age", fill = "Outcome") +
  theme_minimal()

install.packages("dplyr")
library(dplyr)
# Групповые средние значения
# Создание переменной young (TRUE для молодых, FALSE в противном случае)
diabetes_data <- diabetes_data %>%
  mutate(young = if_else(Age < 30, "yes", "no"))
# Вычисление средних значений для двух групп
group_means <- diabetes_data %>%
  group_by(young) %>%
  summarize(mean_Outcome = mean(Outcome), median_Outcome = median(Outcome))
# Упражнение 1: Анализ Pregnancies, Glucose, BloodPressure и т.д.
# ... (аналогично предыдущему коду, просто замените столбцы)
# Например, для Pregnancies и Age:
pregnancies_age_summary <- diabetes_data %>%
  group_by(young) %>%
  summarize(mean_Pregnancies = mean(Pregnancies),
            median_Pregnancies = median(Pregnancies),
            mean_Age = mean(Age),
            median_Age = median(Age))
# Графическое представление
ggplot(diabetes_data, aes(x = young, y = Pregnancies, fill = young)) +
  geom_boxplot() +
  labs(title = "Boxplot of Pregnancies by Age Group", x = "Age Group", y = "Pregnancies") +
  theme_minimal()



library(dplyr)
# Групповые средние значения
# Создание переменной young (TRUE для молодых, FALSE в противном случае)
diabetes_data <- diabetes_data %>%
  mutate(young = if_else(Age < 30, "yes", "no"))
# Вычисление средних значений для двух групп
group_means <- diabetes_data %>%
  group_by(young) %>%
  summarize(mean_Outcome = mean(Outcome), median_Outcome = median(Outcome))
# Анализ каждого столбца
columns_to_analyze <- c("Pregnancies", "Glucose", "BloodPressure", 
                        "SkinThickness", "Insulin", "BMI", 
                        "DiabetesPedigreeFunction", "Age", "Outcome")
for (column in columns_to_analyze) {
  # Сводные статистики для каждой группы (молодые и не молодые)
  summary_stats <- diabetes_data %>%
    group_by(young) %>%
    summarize(
      mean_value = mean(get(column)),
      median_value = median(get(column))
    )
  # Графическое представление
  ggplot(diabetes_data, aes(x = young, y = get(column), fill = young)) +
    geom_boxplot() +
    labs(
      title = paste("Boxplot of", column, "by Age Group"),
      x = "Age Group",
      y = column
    ) +
    theme_minimal()
  # Вывод сводных статистик
  cat(paste("Summary statistics for", column, ":\n"))
  print(summary_stats)
  cat("\n\n")
}


# Трансформация переменной Pregnancies и графическое представление
ggplot(diabetes_data, aes(x = Pregnancies)) +
  geom_density() +
  labs(title = "Density Plot of Pregnancies Variable (Before Transformation)", x = "Pregnancies")
# Трансформация переменной с использованием логарифма
diabetes_data <- diabetes_data %>%
  mutate(log_Pregnancies = log(Pregnancies))
# Повторное графическое представление после трансформации
ggplot(diabetes_data, aes(x = log_Pregnancies)) +
  geom_density() +
  labs(title = "Density Plot of Transformed Pregnancies Variable", x = "log(Pregnancies)")

# Графическое представление переменной Glucose
ggplot(diabetes_data, aes(x = Glucose)) +
  geom_density() +
  labs(title = "Density Plot of Glucose Variable", x = "Glucose")
# Трансформация переменной с использованием логарифма
diabetes_data <- diabetes_data %>%
  mutate(log_Glucose = log(Glucose))
# Повторное графическое представление после трансформации
ggplot(diabetes_data, aes(x = log_Glucose)) +
  geom_density() +
  labs(title = "Density Plot of Transformed Glucose Variable", x = "log(Glucose)")

# Графическое представление переменной BloodPressure
ggplot(diabetes_data, aes(x = BloodPressure)) +
  geom_density() +
  labs(title = "Density Plot of BloodPressure Variable", x = "BloodPressure")
# Трансформация переменной с использованием логарифма
diabetes_data <- diabetes_data %>%
  mutate(log_BloodPressure = log(BloodPressure + 1))  # добавим 1, чтобы избежать логарифма от нуля
# Повторное графическое представление после трансформации
ggplot(diabetes_data, aes(x = log_BloodPressure)) +
  geom_density() +
  labs(title = "Density Plot of Transformed BloodPressure Variable", x = "log(BloodPressure)")

# Графическое представление переменной SkinThickness
ggplot(diabetes_data, aes(x = SkinThickness)) +
  geom_density() +
  labs(title = "Density Plot of SkinThickness Variable", x = "SkinThickness")
# Трансформация переменной с использованием логарифма
diabetes_data <- diabetes_data %>%
  mutate(log_SkinThickness = log(SkinThickness + 1))  # добавим 1, чтобы избежать логарифма от нуля
# Повторное графическое представление после трансформации
ggplot(diabetes_data, aes(x = log_SkinThickness)) +
  geom_density() +
  labs(title = "Density Plot of Transformed SkinThickness Variable", x = "log(SkinThickness)")

# Графическое представление переменной Insulin
ggplot(diabetes_data, aes(x = Insulin)) +
  geom_density() +
  labs(title = "Density Plot of Insulin Variable", x = "Insulin")
# Трансформация переменной с использованием логарифма
diabetes_data <- diabetes_data %>%
  mutate(log_Insulin = log(Insulin + 1))  # добавим 1, чтобы избежать логарифма от нуля
# Повторное графическое представление после трансформации
ggplot(diabetes_data, aes(x = log_Insulin)) +
  geom_density() +
  labs(title = "Density Plot of Transformed Insulin Variable", x = "log(Insulin)")

# Графическое представление переменной BMI
ggplot(diabetes_data, aes(x = BMI)) +
  geom_density() +
  labs(title = "Density Plot of BMI Variable", x = "BMI")
# Трансформация переменной с использованием логарифма
diabetes_data <- diabetes_data %>%
  mutate(log_BMI = log(BMI + 1))  # добавим 1, чтобы избежать логарифма от нуля
# Повторное графическое представление после трансформации
ggplot(diabetes_data, aes(x = log_BMI)) +
  geom_density() +
  labs(title = "Density Plot of Transformed BMI Variable", x = "log(BMI)")

# Графическое представление переменной DiabetesPedigreeFunction
ggplot(diabetes_data, aes(x = DiabetesPedigreeFunction)) +
  geom_density() +
  labs(title = "Density Plot of DiabetesPedigreeFunction Variable", x = "DiabetesPedigreeFunction")
# Трансформация переменной с использованием логарифма
diabetes_data <- diabetes_data %>%
  mutate(log_DiabetesPedigreeFunction = log(DiabetesPedigreeFunction + 1))  # добавим 1, чтобы избежать логарифма от нуля
# Повторное графическое представление после трансформации
ggplot(diabetes_data, aes(x = log_DiabetesPedigreeFunction)) +
  geom_density() +
  labs(title = "Density Plot of Transformed DiabetesPedigreeFunction Variable", x = "log(DiabetesPedigreeFunction)")

# Графическое представление переменной Age
ggplot(diabetes_data, aes(x = Age)) +
  geom_density() +
  labs(title = "Density Plot of Age Variable", x = "Age")
# Трансформация переменной с использованием логарифма
diabetes_data <- diabetes_data %>%
  mutate(log_Age = log(Age + 1))  # добавим 1, чтобы избежать логарифма от нуля
# Повторное графическое представление после трансформации
ggplot(diabetes_data, aes(x = log_Age)) +
  geom_density() +
  labs(title = "Density Plot of Transformed Age Variable", x = "log(Age)")

# Столбчатая диаграмма для переменной Outcome
ggplot(diabetes_data, aes(x = factor(Outcome))) +
  geom_bar() +
  labs(title = "Bar Chart of Outcome Variable", x = "Outcome")

ggplot(data = diabetes_data, aes(x = Age, y = Glucose, color = Outcome)) +
  geom_point() +
  labs(x = "Возраст", y = "Уровень глюкозы", color = "Результат") +
  ggtitle("Связь между возрастом и уровнем глюкозы у молодых людей с диабетом")

ggplot(data = diabetes_data, aes(x = Age, y = BloodPressure, color = Outcome)) +
  geom_point() +
  labs(x = "Возраст", y = "Давление", color = "Результат") +
  ggtitle("Связь между возрастом и давлением у молодых людей с диабетом")

ggplot(data = diabetes_data, aes(x = Age, y = SkinThickness, color = Outcome)) +
  geom_point() +
  labs(x = "Возраст", y = "Толщина кожи", color = "Результат") +
  ggtitle("Связь между возрастом и толщиной кожи у молодых людей с диабетом")

ggplot(data = diabetes_data, aes(x = Age, y = Insulin, color = Outcome)) +
  geom_point() +
  labs(x = "Возраст", y = "Уровень инсулина", color = "Результат") +
  ggtitle("Связь между возрастом и уровнем инсулина у молодых людей с диабетом")

ggplot(data = diabetes_data, aes(x = Age, y = BMI, color = Outcome)) +
  geom_point() +
  labs(x = "Возраст", y = "Индекс массы тела (BMI)", color = "Результат") +
  ggtitle("Связь между возрастом и BMI у молодых людей с диабетом")

ggplot(data = diabetes_data, aes(x = Age, y = DiabetesPedigreeFunction, color = Outcome)) +
  geom_point() +
  labs(x = "Возраст", y = "Diabetes Pedigree Function", color = "Результат") +
  ggtitle("Связь между возрастом и Diabetes Pedigree Function у молодых людей с диабетом")

ggplot(data = diabetes_data, aes(x = Age, fill = as.factor(Outcome))) +
  geom_histogram(binwidth = 5, position = "identity", alpha = 0.7, color = "black") +
  labs(x = "Возраст", y = "Частота", fill = "Результат") +
  ggtitle("Распределение результатов в зависимости от возраста у молодых людей с диабетом")

# Подгрузка данных (замените diabetes_data.csv на ваш набор данных)
data <- read.csv("D:\\dataset\\diabetes2.csv")
# Создание линейной регрессии
model <- lm(Outcome ~ Pregnancies + Glucose + BloodPressure + SkinThickness + Insulin + BMI + DiabetesPedigreeFunction + Age, data = data)
# Вывод сводки результата
summary(model)





data <- read.csv("D:\\dataset\\diabetes2.csv")
model <- lm(Outcome ~ Pregnancies, data = data)
summary(model)
plot(diabetes_data$Pregnancies, diabetes_data$Outcome, main = "Линейная регрессия", xlab = "Pregnancies", ylab = "Outcome")
# Построение линии регрессии
abline(model, col = "black")

data <- read.csv("D:\\dataset\\diabetes2.csv")
model <- lm(Outcome ~ Glucose, data = data)
summary(model)
plot(diabetes_data$Glucose, diabetes_data$Outcome, main = "Линейная регрессия", xlab = "Glucose", ylab = "Outcome")
# Построение линии регрессии
abline(model, col = "black")

data <- read.csv("D:\\dataset\\diabetes2.csv")
model <- lm(Outcome ~ BloodPressure, data = data)
summary(model)
plot(diabetes_data$BloodPressure, diabetes_data$Outcome, main = "Линейная регрессия", xlab = "BloodPressure", ylab = "Outcome")
# Построение линии регрессии
abline(model, col = "black")

data <- read.csv("D:\\dataset\\diabetes2.csv")
model <- lm(Outcome~ SkinThickness, data = data)
summary(model)
plot(diabetes_data$SkinThickness, diabetes_data$Outcome, main = "Линейная регрессия", xlab = "SkinThickness", ylab = "Outcome")
# Построение линии регрессии
abline(model, col = "black")

data <- read.csv("D:\\dataset\\diabetes2.csv")
model <- lm(Outcome ~ Insulin, data = data)
summary(model)
plot(diabetes_data$Insulin, diabetes_data$Outcome, main = "Линейная регрессия", xlab = "Insulin", ylab = "Outcome")
# Построение линии регрессии
abline(model, col = "black")

data <- read.csv("D:\\dataset\\diabetes2.csv")
model <- lm(Outcome ~ BMI, data = data)
summary(model)
plot(diabetes_data$BMI, diabetes_data$Outcome, main = "Линейная регрессия", xlab = "BMI", ylab = "Outcome")
# Построение линии регрессии
abline(model, col = "black")

data <- read.csv("D:\\dataset\\diabetes2.csv")
model <- lm(Outcome ~ DiabetesPedigreeFunction, data = data)
summary(model)
plot(diabetes_data$DiabetesPedigreeFunction, diabetes_data$Outcome, main = "Линейная регрессия", xlab = "DiabetesPedigreeFunction", ylab = "Outcome")
# Построение линии регрессии
abline(model, col = "black")

data <- read.csv("D:\\dataset\\diabetes2.csv")
model <- lm(Outcome ~ Age, data = data)
summary(model)
plot(diabetes_data$Age, diabetes_data$Outcome, main = "Линейная регрессия", xlab = "Age", ylab = "Outcome")
# Построение линии регрессии
abline(model, col = "black")



data <- read.csv("D:\\dataset\\diabetes2.csv")
data <- data[data$Age <= 30, ]
head(data)



data <- read.csv("D:\\dataset\\diabetes2.csv")
data$Pregnancies <- as.factor(data$Pregnancies)
head(data)

install.packages("plotly")
# Загрузка библиотек
library(ggplot2)
library(plotly)

# Загрузка данных
data <- read.csv("D:\\dataset\\diabetes2.csv")
# Построение логистической регрессии для возраста и беременностей
model <- glm(Outcome ~ Age + Pregnancies, data = data, family = "binomial")
# Вывод сводной статистики
summary(model)
# Предсказание вероятностей
data$Probability <- predict(model, type = "response")

# Создание ggplot объекта
plot <- ggplot(data, aes(x = Age, y = Pregnancies, color = factor(Outcome))) +
  geom_point(aes(size = Probability), alpha = 0.7) +
  labs(title = "Влияние наличия беременности и возраста на диабет",
       x = "Age",
       y = "Pregnancies",
       color = "Outcome") +
  scale_color_manual(values = c("red", "blue")) +
  scale_size_continuous(range = c(2, 3))
# Преобразование ggplot объекта в объект plotly
plotly_plot <- ggplotly(plot)
# Вывод интерактивного графика
print(plotly_plot)






# Загрузка библиотек
library(ggplot2)
library(plotly)
# Загрузка данных
data <- read.csv("D:\\dataset\\diabetes2.csv")
# Построение логистической регрессии для возраста и уровня глюкозы
model <- glm(Outcome ~ Age + Glucose, data = data, family = "binomial")
# Вывод сводной статистики
summary(model)
# Предсказание вероятностей
data$Probability <- predict(model, type = "response")
# Создание ggplot объекта
plot <- ggplot(data, aes(x = Age, y = Glucose, color = factor(Outcome))) +
  geom_point(aes(size = Probability), alpha = 0.7) +
  labs(title = "Влияние уровня глюкозы и возраста на диабет",
       x = "Age",
       y = "Glucose",
       color = "Outcome") +
  scale_color_manual(values = c("red", "blue")) +
  scale_size_continuous(range = c(2, 3))
# Преобразование ggplot объекта в объект plotly
plotly_plot <- ggplotly(plot)
# Вывод интерактивного графика
print(plotly_plot)





# Загрузка библиотек
library(ggplot2)
library(plotly)
# Загрузка данных
data <- read.csv("D:\\dataset\\diabetes2.csv")
# Построение логистической регрессии для возраста и артериального давления
model <- glm(Outcome ~ Age + BloodPressure, data = data, family = "binomial")
# Вывод сводной статистики
summary(model)
# Предсказание вероятностей
data$Probability <- predict(model, type = "response")
# Создание ggplot объекта
plot <- ggplot(data, aes(x = Age, y = BloodPressure, color = factor(Outcome))) +
  geom_point(aes(size = Probability), alpha = 0.7) +
  labs(title = "Влияние артериального давления и возраста на диабет",
       x = "Age",
       y = "BloodPressure",
       color = "Outcome") +
  scale_color_manual(values = c("red", "blue")) +
  scale_size_continuous(range = c(2, 2))
# Преобразование ggplot объекта в объект plotly
plotly_plot <- ggplotly(plot)
# Вывод интерактивного графика
print(plotly_plot)








# Загрузка библиотек
library(ggplot2)
library(plotly)
# Загрузка данных
data <- read.csv("D:\\dataset\\diabetes2.csv")
# Построение логистической регрессии для возраста и толщины кожи
model <- glm(Outcome ~ Age + SkinThickness, data = data, family = "binomial")
# Вывод сводной статистики
summary(model)
# Предсказание вероятностей
data$Probability <- predict(model, type = "response")
# Создание ggplot объекта
plot <- ggplot(data, aes(x = Age, y = SkinThickness, color = factor(Outcome))) +
  geom_point(aes(size = Probability), alpha = 0.7) +
  labs(title = "Влияние толщины кожи и возраста на диабет",
       x = "Age",
       y = "SkinThickness",
       color = "Outcome") +
  scale_color_manual(values = c("red", "blue")) +
  scale_size_continuous(range = c(2, 2))
# Преобразование ggplot объекта в объект plotly
plotly_plot <- ggplotly(plot)
# Вывод интерактивного графика
print(plotly_plot)







# Загрузка библиотек
library(ggplot2)
library(plotly)
# Загрузка данных
data <- read.csv("D:\\dataset\\diabetes2.csv")
# Построение логистической регрессии для возраста и уровня инсулина
model <- glm(Outcome ~ Age + Insulin, data = data, family = "binomial")
# Вывод сводной статистики
summary(model)
# Предсказание вероятностей
data$Probability <- predict(model, type = "response")
# Создание ggplot объекта
plot <- ggplot(data, aes(x = Age, y = Insulin, color = factor(Outcome))) +
  geom_point(aes(size = Probability), alpha = 0.7) +
  labs(title = "Влияние уровня инсулина и возраста на диабет",
       x = "Age",
       y = "Insulin",
       color = "Outcome") +
  scale_color_manual(values = c("red", "blue")) +
  scale_size_continuous(range = c(2, 2))

# Преобразование ggplot объекта в объект plotly
plotly_plot <- ggplotly(plot)
# Вывод интерактивного графика
print(plotly_plot)





# Загрузка библиотек
library(ggplot2)
library(plotly)
# Загрузка данных
data <- read.csv("D:\\dataset\\diabetes2.csv")
# Построение логистической регрессии для возраста и индекса массы тела (BMI)
model <- glm(Outcome ~ Age + BMI, data = data, family = "binomial")
# Вывод сводной статистики
summary(model)
# Предсказание вероятностей
data$Probability <- predict(model, type = "response")
# Создание ggplot объекта
plot <- ggplot(data, aes(x = Age, y = BMI, color = factor(Outcome))) +
  geom_point(aes(size = Probability), alpha = 0.7) +
  labs(title = "Влияние индекса массы тела и возраста на диабет",
       x = "Age",
       y = "BMI",
       color = "Outcome") +
  scale_color_manual(values = c("red", "blue")) +
  scale_size_continuous(range = c(2, 2))
# Преобразование ggplot объекта в объект plotly
plotly_plot <- ggplotly(plot)
# Вывод интерактивного графика
print(plotly_plot)




# Загрузка библиотек
library(ggplot2)
library(plotly)
# Загрузка данных
data <- read.csv("D:\\dataset\\diabetes2.csv")
# Построение логистической регрессии для возраста и DiabetesPedigreeFunction
model <- glm(Outcome ~ Age + DiabetesPedigreeFunction, data = data, family = "binomial")
# Вывод сводной статистики
summary(model)
# Предсказание вероятностей
data$Probability <- predict(model, type = "response")
# Создание ggplot объекта
plot <- ggplot(data, aes(x = Age, y = DiabetesPedigreeFunction, color = factor(Outcome))) +
  geom_point(aes(size = Probability), alpha = 0.7) +
  labs(title = "Влияние DiabetesPedigreeFunction и возраста на диабет",
       x = "Age",
       y = "DiabetesPedigreeFunction",
       color = "Outcome") +
  scale_color_manual(values = c("red", "blue")) +
  scale_size_continuous(range = c(2, 2))

# Преобразование ggplot объекта в объект plotly
plotly_plot <- ggplotly(plot)
# Вывод интерактивного графика
print(plotly_plot)



# Установка и подгрузка библиотеки ggplot2
install.packages("ggplot2")
library(ggplot2)

# Важность переменных
coefficients <- coef(model)
standard_errors <- summary(model)$coefficients[, "Std. Error"]

# Расчет важности на основе абсолютных значений коэффициентов
importance <- abs(coefficients) / standard_errors

# Создание датафрейма для графика
importance_df <- data.frame(variable = names(importance), importance = importance)

# Создание барплота
ggplot(importance_df, aes(x = reorder(variable, importance), y = importance)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(title = "Variable Importance in Linear Regression",
       x = "Variable", y = "Importance") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


