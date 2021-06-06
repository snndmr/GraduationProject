import csv
import pickle

import numpy as np
from flask import Flask, request, render_template, flash
from flask_mysqldb import MySQL
from sqlalchemy.dialects import mysql

app = Flask(__name__)
app.secret_key = "cnc"

app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_DB"] = "graduationproject"
app.config["MYSQL_CURSORCLASS"] = "DictCursor"

machine_selected = 1


@app.route('/')
def index():
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM machines")
    return render_template("index.html", data=cursor.fetchall())


@app.route('/machines', methods=["GET", "POST"])
def machines():
    return render_template("add.html")


@app.route('/data', methods=['GET', 'POST'])
def get_data():
    global machine_selected

    csv_data = ''
    with open('static/datasets/{}.csv'.format(machine_selected), newline='') as file:
        reader_object = csv.reader(file, delimiter=',')

        for line in reader_object:
            csv_data += ','.join(line) + '\n'
    return '{}'.format(csv_data)


@app.route('/machine', methods=["GET", "POST"])
def machine():
    if request.method == 'POST':
        global machine_selected
        machine_id = request.form['machine_id']
        machine_model = request.form['machine_model']
        machine_age = request.form['machine_age']
        data = {'machine_id': machine_id, 'machine_model': machine_model, 'machine_age': machine_age}
        machine_selected = machine_id
        return render_template("machine.html", data=[data])


@app.route("/add", methods=["GET", "POST"])
def cnc_add():
    if request.method == 'POST':
        machine_id = float(request.form["machineID"])
        model = float(request.form["model"])
        age = float(request.form["age"])

        cursor = mysql.connection.cursor()
        query = "Insert into machines (machineId, cnc_model, cnc_age) VALUES (%s,%s,%s)"
        cursor.execute(query, (machine_id, model, age))
        mysql.connection.commit()
        cursor.close()
        flash("Hey, {} added.".format(machine_id), "success")
    return render_template("add.html")


@app.route("/cnc", methods=["GET", "POST"])
def cnc():
    data = {'machine_id': '', 'machine_model': '', 'machine_age': ''}

    if request.method == "POST":
        if len(request.form) == 3:
            machine_id = request.form['machine_id']
            machine_model = request.form['cnc_model']
            machine_age = request.form['cnc_age']
            data = {'machine_id': machine_id, 'machine_model': machine_model, 'machine_age': machine_age}
        else:
            registration_number = request.form["registrationNumber"]
            machine_id = float(request.form["machineID"])
            volt = float(request.form["volt"])
            rotate = float(request.form["rotate"])
            pressure = float(request.form["pressure"])
            vibration = float(request.form["vibration"])
            error1count = float(request.form["error1count"])
            error2count = float(request.form["error2count"])
            error3count = float(request.form["error3count"])
            error4count = float(request.form["error4count"])
            error5count = float(request.form["error5count"])
            model = float(request.form["model"])
            age = float(request.form["age"])

            sum_error = error1count + error2count + error3count + error4count + error5count
            new_is_risky = 0 if sum_error == 0 else 1 if sum_error < 5 else 2
            new_age_cat = 1 if age > 15 else 0 if age > 5 else 2
            values = np.array([machine_id, volt, rotate, pressure, vibration, error1count, error2count, error3count,
                               error4count, error5count, model, age, new_age_cat, new_is_risky]).reshape(1, -1)
            log_model = pickle.load(open(r'projects\liftUp_cnc\models\RF.pkl', 'rb'))
            result = log_model.predict(values)

            cursor = mysql.connection.cursor()
            query = "Insert into cnc (cnc_registrationNo,cnc_machineId,cnc_volt,cnc_rotate,cnc_pressure,cnc_vibration," \
                    "cnc_error1count,cnc_error2count,cnc_error3count,cnc_error4count,cnc_error5count,cnc_model,cnc_age," \
                    "cnc_new_age_cat,cnc_new_is_risky,cnc_failure) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            cursor.execute(query, (registration_number, machine_id, volt, rotate, pressure, vibration,
                                   error1count, error2count, error3count, error4count, error5count,
                                   model, age, new_age_cat, new_is_risky, result[0]))
            mysql.connection.commit()
            cursor.close()

            if result == 0:
                flash("Hey, {} operation is success".format(registration_number), "success")
            else:
                flash("Hey, {} operation is unsuccess".format(registration_number), "danger")
    return render_template("cnc.html", data=data)


if __name__ == "__main__":
    mysql = MySQL(app)
    app.run(debug=True)
