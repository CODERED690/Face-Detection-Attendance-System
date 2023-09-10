def attendance():
    list = []
    with open("attendance_data.csv", "r") as file:
        for i in file.readlines():
            hmm = i.split(",")
            list.append({"name": hmm[0], "image": f"/images/User.{hmm[1][1:]}.1.jpg", "date": hmm[2][1:], "time": hmm[3][1:-1]})
    list.reverse()
    return list