<?php
$conn = new mysqli("localhost", "root", "", "website_login");

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $username = $conn->real_escape_string($_POST['username']);
    $email = $conn->real_escape_string($_POST['email']);
    $password = $_POST['password']; 
    $repassword = $_POST['repassword'];

    // Check if passwords match BEFORE hashing
    if ($password !== $repassword) {
        die("Error: Passwords do not match!");
    }

    // Hash the password AFTER verifying it matches
    $hashed_password = password_hash($password, PASSWORD_BCRYPT);

    // Use prepared statement to insert data safely
    $stmt = $conn->prepare("INSERT INTO users (username, email, password) VALUES (?, ?, ?)");
    $stmt->bind_param("sss", $username, $email, $hashed_password);

    if ($stmt->execute()) {
        echo "Registration successful! <a href='index.signin.html'>signin here</a>";
    } else {
        echo "Error: " . $stmt->error;
    }

    $stmt->close();
}

$conn->close();
?>
