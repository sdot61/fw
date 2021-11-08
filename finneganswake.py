# opens finnegans wake txt
# finnegans wake book file downloaded from: https://www.fadedpage.com/showbook.php?pid=20180126
with open("finneganswake.txt", "r") as f:
    text = f.read()

# creates version of text as a list of lines
text_words_orig = text.split('\n')

# creates html using <span> tag
text_list = [f'<span id = {id}>{line}</span><br>' for id, line in enumerate(text_words_orig)]

# joins every element with a space as a string
text = ' '.join(text_list)

# injects finnegans wake text intl html
html_boiler_plate = f'''
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-wEmeIV1mKuiNpC+IOBjI7aAzPcEZeedi5yW5f2yOq55WWLwNGmvvx4Um1vskeMj0" crossorigin="anonymous">

    <title>Finnegans Wake</title>
  </head>
  <body>
    <div class="container">
        <div class="row">
            <h1>Finnegans Wake</h1>
            <div class="col">
            {text}
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0/dist/js/bootstrap.bundle.min.js" integrity="sha384-p34f1UUtsS3wqzfto5wAAmdvj+osOnFyQFpp4Ua3gs/ZVWx6oOypYoCJhGGScy+8" crossorigin="anonymous"></script>
  </body>
</html>
'''
# writes finalized html page for finnegans wake text
with open('templates/finneganswake.html', "w") as f:
    f.write(html_boiler_plate)