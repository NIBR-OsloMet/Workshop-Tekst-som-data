library(tidyverse)
library(tidytext)
library(tidymodels)
library(recipes)
library(textrecipes)
library(caret)
library(readxl)
library(e1071) 
library(rio)
library(ROSE)


## Hente og fikse datasettet ##
df_hatefulle <- read.csv("https://github.com/NIBR-OsloMet/hatefulle-ytringer/raw/main/data/processed_corpus.csv")

  # Endrer variabelnavnene og legger til en ID variabel 
df_hatefulle <- df_hatefulle |>
  rowid_to_column("id") |>
  select(texts = processed_text,
         label = target_label,
         id)

df_hatefulle$label[df_hatefulle$label == 0] <- "hatefull"
df_hatefulle$label[df_hatefulle$label == 1] <- "ikke-hatefull"

  # Konverterer utfallsvariabelen til en faktorvariabel (må være faktor for klassifikasjonen)
df_hatefulle <- df_hatefulle |>
  mutate(label = factor(label))

  # Ta en titt på dataen, og sjekk at alt ser ok ut
glimpse(df_hatefulle)

  # Sjekke balansen
df_hatefulle |>
  count(label)

  # Fikser ubalansen
df_hatefulle_os <- ovun.sample(label~., data = df_hatefulle, method = "both", p = 0.5, seed = 1234)$data

df_hatefulle_os |>
  count(label)

## Modelltrening ##
  # Lager trening, validering og test split
set.seed(1234)
split <- initial_split(df_hatefulle_os, prop = 9/10, strata = label) 
train_val <- training(split)
test <- testing(split)

train_val_split <- initial_split(train_val, prop = 4/5, strata = label)
train <- training(train_val_split)
validate <- testing(train_val_split)

  # Fikser recipe og oppdaterer dataen
hatefulle_recipe <- recipe(label~texts, data=df_hatefulle_os) |>
  step_tokenize(texts) |> # tokeniserer teksten
  step_tokenfilter(texts, max_tokens = 500) |> # beholder kun de mest brukte ordene
  step_stopwords(texts, language = "no") |> # fjerner stoppord
  step_tfidf(texts) |> # kalkulerer tf_idf
  step_normalize(all_predictors()) |> # normaliserer predikatorene
  prep()

new_train <- bake(hatefulle_recipe, new_data = train) 
new_validate <- bake(hatefulle_recipe, new_data = validate)
new_test <- bake(hatefulle_recipe, new_data = test)

  # Trener første modell
set.seed(1234)
svm_rad <- svm(formula = label ~., 
               data = new_train, 
               type = "C-classification",
               kernel = "radial")

train_pred <- new_train |>
  bind_cols(predict(svm_rad, new_data = new_train))

train_pred |>
  accuracy(truth = label, estimate = ...501)

  # Sjekker performance på valideringsdataen
val_rad <- predict(svm_rad, new_validate)
val_rad <- bind_cols(new_validate, val_rad)

val_rad |>
  accuracy(truth = label, estimate = ...501)

new_validate |> 
  bind_cols(predict(svm_rad, new_validate)) |>
  conf_mat(truth = label, estimate = ...501) |>
  autoplot(type = 'heatmap')

cm <- confusionMatrix(table(val_rad$...501, new_validate$label), positive = "hatefull") 
  # Legg merke til at vi må presisere hvilken kategori som er den positive, for å sørge for at målene blir kalkulert riktig
cm$byClass["Precision"]
cm$byClass["Recall"]
cm$byClass["F1"]

  # Parameter tuning
start_time <- Sys.time() # sys.time for å sjekke for lang tid kodesnutten tar å kjøre
set.seed(1234)
svm_tune <- tune.svm(x = label ~., 
                     data = new_train,
                     gamma = seq(0, 1, by = 0.1), 
                     cost = seq(1, 3, by = 1), 
                     kernel = "radial")
end_time <- Sys.time()
end_time - start_time

svm_tune$best.parameters$gamma
svm_tune$best.parameters$cost

  # Trener ny modell med de nye parameterne
set.seed(1234)
svm_model <- svm(formula = label ~ .,
                 data = new_train,
                 type = 'C-classification',
                 kernel = 'radial',
                 cost = svm_tune$best.parameters$cost, 
                 gamma = svm_tune$best.parameters$gamma)

validate_tune <- predict(svm_model, new_validate)
validate_tune <- bind_cols(new_validate, validate_tune)

cm_tuned <- confusionMatrix(table(validate_tune$...501, new_validate$label), positive = "hatefull") 
cm_tuned$byClass["Precision"]
cm_tuned$byClass["Recall"]
cm_tuned$byClass["F1"]

new_validate |>
  bind_cols(predict(svm_model, new_validate)) |>
  conf_mat(truth = label, estimate = ...501) |>
  autoplot(type = 'heatmap')

  # Repeat
start_time <- Sys.time() 
set.seed(1234)
svm_tune_2 <- tune.svm(x = label ~., 
                       data = new_train,
                       gamma = 10^(-4:-1), 
                       cost = seq(0.01, 1, by = 0.1), 
                       kernel = "radial")
end_time <- Sys.time()
end_time - start_time

svm_tune_2$best.parameters$gamma
svm_tune_2$best.parameters$cost

set.seed(1234)
svm_model_2 <- svm(formula = label ~ .,
                   data = new_train,
                   type = 'C-classification',
                   kernel = 'radial',
                   cost = 0.91, 
                   gamma = 0.01)

validate_tune_2 <- predict(svm_model_2, new_validate)
validate_tune_2 <- bind_cols(new_validate, validate_tune_2)

cm_tuned_2 <- confusionMatrix(table(validate_tune_2$...501, new_validate$label), positive = "hatefull") 
cm_tuned_2$byClass["Precision"]
cm_tuned_2$byClass["Recall"]
cm_tuned_2$byClass["F1"]

new_validate |>
  bind_cols(predict(svm_model_2, new_validate)) |>
  conf_mat(truth = label, estimate = ...501) |>
  autoplot(type = 'heatmap')

## Bruker deen endelige modellen på testdataen
test_pred <- predict(svm_model_2, new_test)
test_pred <- bind_cols(new_test, test_pred)

cm_test <- confusionMatrix(table(test_pred$...501, new_test$label), positive = "hatefull") 
cm_test$byClass["Precision"]
cm_test$byClass["Recall"]
cm_test$byClass["F1"]

new_test |>
  bind_cols(predict(svm_model_2, new_test)) |>
  conf_mat(truth = label, estimate = ...501) |>
  autoplot(type = 'heatmap')